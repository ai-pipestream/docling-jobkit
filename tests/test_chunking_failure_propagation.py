"""A chunker or tokenizer failure must surface as an error, not empty chunks."""

from pathlib import Path
from unittest.mock import patch

import pytest

from docling.datamodel.base_models import ConversionStatus, DoclingComponentType
from docling.datamodel.service.responses import FailureCategory, FailurePhase
from docling.datamodel.service.targets import InBodyTarget
from docling_core.types.doc.document import DoclingDocument

from docling_jobkit.convert.chunking import (
    DocumentChunkerManager,
    process_chunkable_results,
)
from docling_jobkit.datamodel.chunking import HybridChunkerOptions
from docling_jobkit.datamodel.exportable_document import ExportableDocument
from docling_jobkit.datamodel.result import ChunkedDocumentResult
from docling_jobkit.datamodel.task import Task
from docling_jobkit.datamodel.task_meta import TaskType
from docling_jobkit.public_errors import (
    INTERNAL_TASK_ERROR_MESSAGE,
    ChunkingError,
    build_public_task_error,
    chunking_error_message,
    classify_public_task_failure,
)


class _RaisingChunkerManager:
    """Stands in for DocumentChunkerManager; raises for the named files."""

    def __init__(self, failing: set[str], error: Exception):
        self.failing = failing
        self.error = error

    def chunk_document(self, *, document, filename, options):
        if filename in self.failing:
            raise self.error
        return iter(())


def _task() -> Task:
    return Task(
        task_id="chunk-failure",
        task_type=TaskType.CHUNK,
        chunking_options=HybridChunkerOptions(),
        target=InBodyTarget(),
    )


def _doc(name: str) -> ExportableDocument:
    return ExportableDocument(
        file=Path(name),
        status=ConversionStatus.SUCCESS,
        document=DoclingDocument(name=Path(name).stem),
    )


def test_chunker_failure_for_every_document_raises_chunking_error(tmp_path: Path):
    manager = _RaisingChunkerManager(
        {"a.pdf"}, OSError("401 Client Error: tokenizer fetch rejected")
    )
    with pytest.raises(ChunkingError) as excinfo:
        process_chunkable_results(
            task=_task(),
            exportable_documents=[_doc("a.pdf")],
            work_dir=tmp_path,
            chunker_manager=manager,  # type: ignore[arg-type]
        )
    assert str(excinfo.value) == "Document chunking failed (OSError)."
    # The raw exception text (which can carry URLs or hints of credentials) is
    # not part of the client-facing message.
    assert "401" not in str(excinfo.value)


def test_chunking_error_is_classified_as_typed_backend_failure():
    exc = ChunkingError(chunking_error_message(["OSError"]))
    failure = classify_public_task_failure(exc, task_id="t")
    assert failure.category == FailureCategory.BACKEND_FAILURE
    assert failure.phase == FailurePhase.EXECUTION
    assert failure.retryable is False
    assert failure.message == "Document chunking failed (OSError)."
    assert build_public_task_error(exc) == "Document chunking failed (OSError)."
    # Unrelated exceptions keep the sanitized internal message.
    assert build_public_task_error(OSError("x")) == INTERNAL_TASK_ERROR_MESSAGE


def test_chunking_error_keeps_its_type_when_wrapped_or_chained():
    message = "Document chunking failed (ValueError)."
    try:
        try:
            raise ChunkingError(message)
        except ChunkingError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        failure = classify_public_task_failure(wrapped, task_id="t")
    assert failure.category == FailureCategory.BACKEND_FAILURE
    assert failure.message == message

    # A ChunkingError raised from the original exception is not unwrapped to it.
    try:
        try:
            raise ValueError("raw detail")
        except ValueError as cause:
            raise ChunkingError(message) from cause
    except ChunkingError as chained:
        failure = classify_public_task_failure(chained, task_id="t")
    assert failure.category == FailureCategory.BACKEND_FAILURE
    assert failure.message == message


def test_chunking_error_message_is_sorted_and_deduplicated():
    assert (
        chunking_error_message(["OSError", "ValueError", "OSError"])
        == "Document chunking failed (OSError, ValueError)."
    )


def test_partial_chunker_failure_marks_only_that_document_failed(tmp_path: Path):
    manager = _RaisingChunkerManager({"bad.pdf"}, ValueError("boom"))
    task_result = process_chunkable_results(
        task=_task(),
        exportable_documents=[_doc("good.pdf"), _doc("bad.pdf")],
        work_dir=tmp_path,
        chunker_manager=manager,  # type: ignore[arg-type]
    )
    assert isinstance(task_result.result, ChunkedDocumentResult)
    assert task_result.num_succeeded == 1
    assert task_result.num_failed == 1
    good, bad = task_result.result.documents
    assert good.status == ConversionStatus.SUCCESS
    assert good.errors == []
    assert bad.status == ConversionStatus.FAILURE
    assert len(bad.errors) == 1
    error = bad.errors[0]
    assert error.component_type == DoclingComponentType.PIPELINE
    assert error.module_name == "ValueError"
    assert error.error_message == "Document chunking failed (ValueError)."
    assert error.category == FailureCategory.BACKEND_FAILURE


def test_conversion_failure_without_chunker_error_keeps_old_behavior(tmp_path: Path):
    failed = ExportableDocument(
        file=Path("file.pdf"),
        status=ConversionStatus.FAILURE,
        errors=[],
        timings={},
    )
    task_result = process_chunkable_results(
        task=_task(),
        exportable_documents=[failed],
        work_dir=tmp_path,
        chunker_manager=_RaisingChunkerManager(set(), RuntimeError("unused")),  # type: ignore[arg-type]
    )
    assert isinstance(task_result.result, ChunkedDocumentResult)
    assert task_result.result.documents[0].status == ConversionStatus.FAILURE
    assert task_result.num_failed == 1


def test_tokenizer_load_failure_in_real_chunker_manager_raises(tmp_path: Path):
    """Force the real HF tokenizer load to fail (expired HF_TOKEN gives a 401)."""
    manager = DocumentChunkerManager()
    with patch(
        "docling_jobkit.convert.chunking.HuggingFaceTokenizer.from_pretrained",
        side_effect=OSError("401 Client Error: Unauthorized for url"),
    ) as from_pretrained:
        with pytest.raises(ChunkingError) as excinfo:
            process_chunkable_results(
                task=_task(),
                exportable_documents=[_doc("report.pdf")],
                work_dir=tmp_path,
                chunker_manager=manager,
            )
    assert from_pretrained.called
    assert str(excinfo.value) == "Document chunking failed (RuntimeError)."
