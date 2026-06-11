"""Tests for the convert/export decoupling (PrefetchIterator + results paths)."""

import base64
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.service.sources import FileSource
from docling.datamodel.service.targets import InBodyTarget
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DoclingDocument

from docling_jobkit.convert.prefetch import PrefetchIterator
from docling_jobkit.convert.results import process_exportable_results
from docling_jobkit.datamodel.convert import ConvertDocumentsOptions
from docling_jobkit.datamodel.exportable_document import ExportableDocument
from docling_jobkit.datamodel.result import DocumentResultItem, ZipArchiveResult
from docling_jobkit.datamodel.task import Task

# ---------------------------------------------------------------------------
# PrefetchIterator unit tests
# ---------------------------------------------------------------------------


def test_prefetch_yields_all_items_in_order():
    items = list(range(200))

    with PrefetchIterator(iter(items), buffer_size=1) as prefetched:
        assert list(prefetched) == items


def test_prefetch_empty_source():
    with PrefetchIterator(iter(()), buffer_size=1) as prefetched:
        assert list(prefetched) == []


def test_prefetch_rejects_invalid_buffer_size():
    with pytest.raises(ValueError, match="buffer_size"):
        PrefetchIterator(iter(()), buffer_size=0)


def test_prefetch_exception_propagates_after_prior_items():
    def source() -> Iterator[int]:
        yield 1
        yield 2
        raise RuntimeError("conversion blew up")

    with PrefetchIterator(source(), buffer_size=1) as prefetched:
        assert next(prefetched) == 1
        assert next(prefetched) == 2
        with pytest.raises(RuntimeError, match="conversion blew up"):
            next(prefetched)
        with pytest.raises(StopIteration):
            next(prefetched)


def test_prefetch_produces_ahead_of_consumer():
    """The producer must pull the next item while the consumer holds one."""
    second_item_pulled = threading.Event()

    def source() -> Iterator[str]:
        yield "first"
        second_item_pulled.set()
        yield "second"

    with PrefetchIterator(source(), buffer_size=1) as prefetched:
        assert next(prefetched) == "first"
        # Without any further consumption, the producer should already be
        # working on the next item: this is the conversion/export overlap.
        assert second_item_pulled.wait(timeout=5.0)
        assert next(prefetched) == "second"


def test_prefetch_respects_buffer_bound():
    produced: list[int] = []

    def source() -> Iterator[int]:
        for i in range(50):
            produced.append(i)
            yield i

    with PrefetchIterator(source(), buffer_size=1) as prefetched:
        # Consume a single item to start the (lazy) producer, then stop
        # consuming. The producer may keep one item buffered and hold one
        # more while waiting for buffer space, so it can run at most
        # consumed + buffer_size + 1 items ahead.
        assert next(prefetched) == 0
        deadline = time.monotonic() + 2.0
        while len(produced) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)
        assert len(produced) == 3

        assert list(prefetched) == list(range(1, 50))
        assert len(produced) == 50


def test_prefetch_close_stops_production():
    produced: list[int] = []

    def source() -> Iterator[int]:
        for i in range(1000):
            produced.append(i)
            yield i

    prefetched = PrefetchIterator(source(), buffer_size=1)
    assert next(prefetched) == 0
    prefetched.close()

    # close() joins the producer, so production has stopped for good.
    assert len(produced) <= 4
    count_after_close = len(produced)
    time.sleep(0.2)
    assert len(produced) == count_after_close

    with pytest.raises(StopIteration):
        next(prefetched)


def test_prefetch_close_is_idempotent():
    prefetched = PrefetchIterator(iter(range(5)), buffer_size=1)
    assert next(prefetched) == 0
    prefetched.close()
    prefetched.close()


def test_prefetch_lazy_start_runs_nothing_when_unconsumed():
    produced: list[int] = []

    def source() -> Iterator[int]:
        produced.append(0)
        yield 0

    prefetched = PrefetchIterator(source(), buffer_size=1)
    prefetched.close()
    assert produced == []


def test_prefetch_close_finalizes_generator_source():
    cleaned_up = threading.Event()

    def source() -> Iterator[int]:
        try:
            for i in range(1000):
                yield i
        finally:
            cleaned_up.set()

    prefetched = PrefetchIterator(source(), buffer_size=1)
    assert next(prefetched) == 0
    prefetched.close()
    assert cleaned_up.is_set()


# ---------------------------------------------------------------------------
# process_exportable_results integration (in-body / zip paths)
# ---------------------------------------------------------------------------


class _FakeDoc(DoclingDocument):
    markdown_image_modes: ClassVar[list[ImageRefMode]] = []

    def save_as_json(self, filename, image_mode, artifacts_dir):
        del image_mode
        Path(filename).write_text('{"ok": true}', encoding="utf-8")
        artifact_path = Path(filename).parent / artifacts_dir
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "figure.png").write_bytes(b"png")

    def export_to_markdown(self, image_mode=ImageRefMode.PLACEHOLDER, **_kwargs):
        self.__class__.markdown_image_modes.append(image_mode)
        return "converted markdown"


def _make_exportable_document(
    *,
    filename: str = "paper.pdf",
    status: ConversionStatus = ConversionStatus.SUCCESS,
) -> ExportableDocument:
    return ExportableDocument(
        file=Path(filename),
        status=status,
        document=(
            _FakeDoc.model_construct(pages={}, tables=[], pictures=[])
            if status != ConversionStatus.FAILURE
            else None
        ),
    )


def _make_in_body_task() -> Task:
    return Task(
        task_id="task-inbody",
        sources=[
            FileSource(
                base64_string=base64.b64encode(b"fake-pdf").decode(),
                filename="paper.pdf",
            )
        ],
        target=InBodyTarget(),
        convert_options=ConvertDocumentsOptions(to_formats=[OutputFormat.JSON]),
    )


def test_single_document_in_body_result(tmp_path: Path):
    task_result = process_exportable_results(
        task=_make_in_body_task(),
        exportable_documents=iter([_make_exportable_document()]),
        work_dir=tmp_path,
    )

    assert isinstance(task_result.result, DocumentResultItem)
    assert task_result.result.status == ConversionStatus.SUCCESS
    assert task_result.result.document.json_content is not None
    assert task_result.num_converted == 1
    assert task_result.num_succeeded == 1
    assert task_result.num_failed == 0


def test_multi_document_in_body_returns_zip_archive(tmp_path: Path):
    documents = [
        _make_exportable_document(filename="a.pdf"),
        _make_exportable_document(filename="b.pdf"),
        _make_exportable_document(filename="c.pdf"),
    ]

    task_result = process_exportable_results(
        task=_make_in_body_task(),
        exportable_documents=iter(documents),
        work_dir=tmp_path,
    )

    assert isinstance(task_result.result, ZipArchiveResult)
    assert len(task_result.result.content) > 0
    assert task_result.num_converted == 3
    assert task_result.num_succeeded == 3
    assert task_result.num_failed == 0


def test_multi_document_path_releases_document_references(tmp_path: Path):
    documents = [
        _make_exportable_document(filename="a.pdf"),
        _make_exportable_document(filename="b.pdf"),
    ]

    process_exportable_results(
        task=_make_in_body_task(),
        exportable_documents=iter(documents),
        work_dir=tmp_path,
    )

    # Streaming export must drop document payloads as soon as each document
    # has been materialized, keeping peak memory bounded.
    assert all(doc.document is None for doc in documents)


def test_multi_document_counts_failed_documents(tmp_path: Path):
    documents = [
        _make_exportable_document(filename="good.pdf"),
        _make_exportable_document(filename="bad.pdf", status=ConversionStatus.FAILURE),
    ]

    task_result = process_exportable_results(
        task=_make_in_body_task(),
        exportable_documents=iter(documents),
        work_dir=tmp_path,
    )

    assert isinstance(task_result.result, ZipArchiveResult)
    assert task_result.num_converted == 2
    assert task_result.num_succeeded == 1
    assert task_result.num_failed == 1


def test_no_documents_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No documents were generated"):
        process_exportable_results(
            task=_make_in_body_task(),
            exportable_documents=iter(()),
            work_dir=tmp_path,
        )


def test_conversion_error_mid_stream_propagates(tmp_path: Path):
    def documents() -> Iterator[ExportableDocument]:
        yield _make_exportable_document(filename="ok.pdf")
        raise RuntimeError("converter crashed")

    with pytest.raises(RuntimeError, match="converter crashed"):
        process_exportable_results(
            task=_make_in_body_task(),
            exportable_documents=documents(),
            work_dir=tmp_path,
        )


def test_export_overlaps_with_conversion(tmp_path: Path):
    """While document one is exported, later documents must keep converting.

    With strictly serial processing, the source generator only advances when
    the consumer pulls the next item, so the export of document one would
    deadlock waiting for document three and the test would fail on timeout.
    """
    third_doc_converted = threading.Event()

    class _BlockingDoc(_FakeDoc):
        def save_as_json(self, filename, image_mode, artifacts_dir):
            assert third_doc_converted.wait(timeout=10.0), (
                "conversion did not overlap with export"
            )
            super().save_as_json(filename, image_mode, artifacts_dir)

    def documents() -> Iterator[ExportableDocument]:
        first = _make_exportable_document(filename="a.pdf")
        first.document = _BlockingDoc.model_construct(pages={}, tables=[], pictures=[])
        yield first
        yield _make_exportable_document(filename="b.pdf")
        third_doc_converted.set()
        yield _make_exportable_document(filename="c.pdf")

    task_result = process_exportable_results(
        task=_make_in_body_task(),
        exportable_documents=documents(),
        work_dir=tmp_path,
    )
    assert task_result.num_converted == 3
    assert task_result.num_succeeded == 3
