"""Bounded background prefetching for lazy conversion result iterators.

Document conversion and result export are CPU/GPU phases that alternate when a
lazy conversion iterator is consumed inline: the converter sits idle while a
finished document is exported, and the exporter sits idle while the next
document converts. :class:`PrefetchIterator` decouples the two by driving the
source iterator on a dedicated producer thread, buffering a bounded number of
items, so conversion of document *N+1* overlaps with export of document *N*.

The buffer is intentionally small: a single in-flight ``ConversionResult`` can
hold every page image of a document, so each buffered item may account for a
significant amount of memory.
"""

import enum
import logging
import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


class _MsgKind(enum.Enum):
    ITEM = enum.auto()
    DONE = enum.auto()
    ERROR = enum.auto()


# How long a blocked producer waits on the buffer before re-checking for
# cancellation. Only affects shutdown latency, not throughput.
_PUT_POLL_INTERVAL_SECONDS = 0.1


class PrefetchIterator(Generic[T]):
    """Iterate *source* on a background thread through a bounded buffer.

    Items are yielded strictly in source order. An exception raised by the
    source is re-raised to the consumer at the position where the source
    failed, after all previously produced items have been delivered.

    The source iterator is driven by exactly one thread (the producer), so a
    source that is not thread-safe — such as a generator wrapping a
    ``DocumentConverter`` — remains single-threaded; only *which* thread runs
    it changes.

    Instances are context managers; :meth:`close` is idempotent and must be
    called (or the ``with`` block exited) to guarantee the producer thread has
    terminated before the underlying source is reused elsewhere.

    Args:
        source: The iterable to consume on the producer thread.
        buffer_size: Maximum number of items buffered ahead of the consumer.
            Must be at least 1. In addition to the buffered items, one item
            may be held by the producer while it waits for buffer space.
    """

    def __init__(self, source: Iterable[T], buffer_size: int = 1) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self._queue: queue.Queue[tuple[_MsgKind, Any]] = queue.Queue(
            maxsize=buffer_size
        )
        self._cancelled = threading.Event()
        self._finished = False
        # The producer starts lazily on first consumption so that constructing
        # the iterator never triggers work on the source: callers may still
        # fail fast (e.g. on target validation) before anything is converted.
        self._thread: threading.Thread | None = None
        self._source: Iterable[T] | None = source

    def _ensure_started(self) -> None:
        if self._thread is not None or self._finished:
            return
        source, self._source = self._source, None
        assert source is not None
        self._thread = threading.Thread(
            target=self._produce,
            args=(source,),
            name="docling-export-prefetch",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------- producer
    def _produce(self, source: Iterable[T]) -> None:
        source_iter = iter(source)
        try:
            for item in source_iter:
                if self._cancelled.is_set() or not self._offer((_MsgKind.ITEM, item)):
                    return
            self._offer((_MsgKind.DONE, None))
        # BaseException: even KeyboardInterrupt/SystemExit raised by the source
        # must surface on the consumer side rather than die with this thread.
        except BaseException as exc:
            if not self._offer((_MsgKind.ERROR, exc)):
                _log.warning(
                    "Prefetch source raised after cancellation; error is dropped.",
                    exc_info=exc,
                )
        finally:
            # Close generator sources on the same thread that ran them so any
            # finally blocks (resource cleanup) execute deterministically.
            close_source = getattr(source_iter, "close", None)
            if close_source is not None:
                try:
                    close_source()
                except Exception:
                    _log.exception("Error while closing prefetch source iterator.")

    def _offer(self, msg: tuple[_MsgKind, Any]) -> bool:
        """Put *msg* on the buffer, returning False if cancelled while waiting."""
        while not self._cancelled.is_set():
            try:
                self._queue.put(msg, timeout=_PUT_POLL_INTERVAL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    # ------------------------------------------------------------- consumer
    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._finished:
            raise StopIteration
        self._ensure_started()
        kind, payload = self._queue.get()
        if kind is _MsgKind.ITEM:
            return payload  # type: ignore[no-any-return]
        self._finished = True
        # Producer has exited (DONE or ERROR); reap the thread.
        assert self._thread is not None
        self._thread.join()
        if kind is _MsgKind.ERROR:
            raise payload
        raise StopIteration

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Stop the producer and wait for it to terminate.

        The join is deliberately unbounded: the producer is the only thread
        driving the underlying source (typically a converter), and returning
        while it is still converting would allow a subsequent task to use the
        converter concurrently. At worst this waits for one in-flight document
        to finish converting — the same bound as fully serial processing.
        """
        self._finished = True
        self._cancelled.set()
        self._source = None
        if self._thread is None:
            return
        # Drain buffered items so a producer blocked on a full buffer can
        # observe the cancellation promptly.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join()

    def __enter__(self) -> "PrefetchIterator[T]":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
