"""Off-thread durability for donor captures.

vLLM calls ``save_kv_layer`` from inside the forward pass, so everything that
happens there is time the request's own prefill did not spend prefilling. The
per-layer device-to-host copy has to stay on that thread -- it reads the paged
KV cache the forward pass owns -- but nothing after it does. The safetensors
write, the donor's metadata record and the directory work behind them are
handed to this writer instead. What the writer does not finish before the step
ends is paid at the join in ``wait_for_save``, which is still inside
``execute_model``: the write is overlapped with the rest of the forward pass,
not removed from the step.

Ordering is as much the point as the deferral. The queue is FIFO with a single
consumer, so jobs land in submission order, and the connector submits a
donor's metadata record only after every layer job for that donor. The
metadata record is what makes a donor discoverable, so a donor is discoverable
only once its tensors are on disk.

The queue is bounded. A full queue means the store is slower than the engine
is producing captures, and the submitting thread blocks until there is room:
a dropped job would leave a donor whose metadata promises layers that were
never written, which is the one outcome this module exists to prevent. The
block is reported to the caller so it is counted rather than merely felt.

Teardown is bounded too, and that bound is the only thing standing between a
wedged store and a process that will not exit. ``close`` spends at most its
timeout in total -- on the sentinel and on the join together -- because a
store that has stopped draining the queue is exactly the case where the
sentinel cannot be enqueued either. The thread is a daemon on top of that, and
the interpreter's ``atexit`` hook closes whatever is still live. Whatever that
deadline leaves in the queue is handed back as a failed write rather than left
outstanding: an abandoned job is counted, and no later join waits on it.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("semblend_vllm_connector")

#: What a job is: one layer's host copy, or one donor's metadata record.
LAYER = "layer"
METADATA = "metadata"

#: How long ``close`` waits for a drain when the caller names no timeout.
DEFAULT_CLOSE_TIMEOUT_S = 5.0

#: Every writer with a live thread, weakly held. A writer dropped without a
#: close is collected rather than pinned by the exit hook, and the hook itself
#: is registered once for the process instead of once per writer.
_LIVE_WRITERS: "weakref.WeakSet[CaptureWriter]" = weakref.WeakSet()
_LIVE_WRITERS_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def close_live_writers() -> None:
    """Close every writer that still has a thread. Registered with ``atexit``.

    Also the teardown hook the suite uses: a test that builds a connector and
    never shuts it down would otherwise leave a thread and a queue of host
    tensors behind for the rest of the session.
    """
    with _LIVE_WRITERS_LOCK:
        writers = list(_LIVE_WRITERS)
    for writer in writers:
        try:
            writer.close()
        except Exception:  # teardown must not raise out of an exit hook
            logger.exception("SemBlend capture writer close failed at teardown")


@dataclass(frozen=True)
class WriteJob:
    """One unit of durable work, carrying everything the write needs.

    ``payload`` is the host tensor for a layer job and is ignored for a
    metadata job; ``token_count`` is the donor length the record publishes.
    ``layer_names`` is the other way round: the layers a metadata job is about
    to vouch for, so the publish can check they are still there.
    """

    request_id: str
    namespace: str
    kind: str
    layer_name: str | None = None
    payload: Any = None
    token_count: int = 0
    layer_names: tuple[str, ...] = ()
    #: For a layer job whose host copy is still in flight on a side stream:
    #: the CUDA event that completes it. The writer waits on it, the forward
    #: pass never does.
    ready: Any = None


class CaptureWriter:
    """A single background thread draining a bounded FIFO of write jobs.

    The thread is started on the first submission, so a connector role that
    never captures anything never owns one. It is a daemon thread with an
    ``atexit`` shutdown on top: the ``atexit`` hook gives a healthy queue the
    chance to drain before the interpreter goes away, and every wait inside
    that hook is bounded, so a wedged store delays the exit by the close
    timeout rather than preventing it.
    """

    def __init__(
        self,
        *,
        write: Callable[[WriteJob], None],
        on_error: Callable[[WriteJob, BaseException], None],
        max_queue_depth: int,
        on_blocked: Callable[[WriteJob, float], None] | None = None,
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_S,
        thread_name: str = "semblend-capture-writer",
    ) -> None:
        self._write = write
        self._on_error = on_error
        self._on_blocked = on_blocked
        self._close_timeout = max(0.0, float(close_timeout))
        self._queue: "queue.Queue[WriteJob | None]" = queue.Queue(
            maxsize=max(1, int(max_queue_depth))
        )
        self._thread: threading.Thread | None = None
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._closed = False
        # Set only when the sentinel could not be enqueued: the thread then
        # stops at the next job boundary instead of waiting for a sentinel
        # that is never coming.
        self._stopping = False

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def submit(self, job: WriteJob, *, timeout: float | None = None) -> None:
        """Hand one job to the writer, blocking only if the queue is full.

        ``timeout`` bounds that wait and raises ``queue.Full`` instead, for the
        one caller that must not block on a full queue: teardown, where the
        queue is full precisely because the store has stopped draining it. The
        job is then a failed write, which is a named outcome, rather than a
        process that will not exit.
        """
        self._ensure_thread()
        try:
            self._queue.put_nowait(job)
            return
        except queue.Full:
            pass
        started = time.monotonic()
        if timeout is None:
            # Blocking, deliberately: the alternative to waiting is a donor
            # that advertises layers nobody wrote.
            self._queue.put(job)
        else:
            self._queue.put(job, timeout=max(0.0, float(timeout)))
        if self._on_blocked is not None:
            self._on_blocked(job, (time.monotonic() - started) * 1000.0)

    def join(self) -> None:
        """Return once every job submitted so far has been written.

        A closed writer never waits. ``close`` may have abandoned jobs on a
        store that stopped draining, and vLLM calls ``wait_for_save`` after
        every ``save_kv_layer`` -- including the ones that come after a
        shutdown, which are named write failures -- so a join that waited on
        work nobody will ever run would hang the step instead of failing it.
        """
        if self._thread is None or self._closed:
            return
        self._queue.join()

    def close(self, timeout: float | None = None) -> None:
        """Drain the queue and stop the thread. Safe to call more than once.

        One deadline covers the whole call. Enqueuing the sentinel is part of
        the wait rather than ahead of it: a store that has stopped draining
        leaves the queue full, and an unbounded put there would block forever
        with the timeout below never reached.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        budget = self._close_timeout if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + budget
        queued = self._queue.qsize()
        if queued:
            logger.info(
                "SemBlend capture writer draining %d queued donor write(s), up to %.1fs",
                queued,
                budget,
            )
        try:
            # The sentinel goes through the same FIFO, so everything already
            # queued is written before the thread stops.
            self._queue.put(None, timeout=max(0.0, deadline - time.monotonic()))
        except queue.Full:
            # Nothing is draining the queue, so the thread is wedged in a
            # write. Tell it to stop if it ever comes back, hand back every
            # job it will never reach, and stop waiting.
            self._stopping = True
            self._abandon_queued()
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            logger.warning(
                "SemBlend capture writer did not stop within %.1fs; "
                "queued donor writes may be incomplete",
                budget,
            )
        else:
            with _LIVE_WRITERS_LOCK:
                _LIVE_WRITERS.discard(self)

    def _abandon_queued(self) -> None:
        """Fail every job still queued, so nothing waits on it afterwards.

        Called only when ``close`` gave up on the sentinel, which is the case
        where the store has stopped draining the queue. Each abandoned job is
        reported through ``on_error``: it leaves its donor without a metadata
        record, which is the outcome a failed write already has a name for,
        and reporting it is the difference between a counted abandonment and
        a silent one. Marking them done is what keeps the queue's own counter
        from stranding a later join on work that will never run.
        """
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if job is not None:
                    self._on_error(
                        job, RuntimeError("capture writer closed with this write still queued")
                    )
            except Exception:
                logger.exception("SemBlend capture write error handler failed")
            finally:
                self._queue.task_done()

    def _ensure_thread(self) -> None:
        global _ATEXIT_REGISTERED
        with self._lock:
            if self._closed:
                raise RuntimeError("capture writer is closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
            self._thread.start()
        with _LIVE_WRITERS_LOCK:
            _LIVE_WRITERS.add(self)
            if not _ATEXIT_REGISTERED:
                atexit.register(close_live_writers)
                _ATEXIT_REGISTERED = True

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._write(job)
            except Exception as exc:  # one bad job must not stop the writer
                try:
                    self._on_error(job, exc)
                except Exception:
                    logger.exception("SemBlend capture write error handler failed")
            finally:
                self._queue.task_done()
            if self._stopping:
                # close() gave up on the sentinel. Anything still queued is
                # abandoned: its donor has no metadata record, so it is not
                # discoverable and cannot be loaded from.
                return
