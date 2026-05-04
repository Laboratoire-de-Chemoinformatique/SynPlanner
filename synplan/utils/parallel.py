"""Generic parallelization utilities.

This module provides small, reusable helpers for ProcessPool-based
parallel mapping with backpressure. Keep format-specific I/O utilities in
``synplan.utils.files``.

Key components:
- ``process_pool_map_stream``: lazy parallel map with backpressure, timeout,
  ordered mode, and worker initializer support.
- ``graceful_shutdown``: context manager for SIGTERM/SIGINT handling.
- ``select_device``, ``default_num_workers``: hardware helpers.
"""

import contextlib
import logging
import os
import signal
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CANCEL_FUTURES = sys.version_info >= (3, 9)
_RECYCLE_WORKERS = sys.version_info >= (3, 11)
_FORCE_TERMINATE_ATTR = "_synplan_force_terminate_workers"
_PROCESS_EXIT_TIMEOUT = 2.0
_TERMINATE_TIMEOUT = 2.0
_KILL_TIMEOUT = 1.0
_MANAGER_THREAD_TIMEOUT = 1.0


def select_device(device: str | None = None) -> torch.device:
    """Auto-detect best available device: cuda > mps > cpu.

    Parameters
    ----------
    device
        Explicit device string (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
        When *None* the best available accelerator is chosen automatically.

    Returns
    -------
    torch.device
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers(cap: int = 8) -> int:
    """Return a sensible default worker count, capped at *cap*."""
    return min(os.cpu_count() or 4, cap)


def process_pool_map_stream(
    items: Iterable[Any],
    worker_fn: Callable[[Any], Any],
    *,
    max_workers: int,
    max_pending: int | None = None,
    timeout: float = 300.0,
    ordered: bool = False,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    max_tasks_per_child: int | None = None,
    on_timeout: Callable[[FuturesTimeoutError, Any], Any] | None = None,
) -> Iterator[Any]:
    """Submit tasks lazily and yield results with backpressure.

    Parameters
    ----------
    items
        Iterable of inputs. Each item is passed to ``worker_fn``.
    worker_fn
        Top-level module function (must be picklable). Receives one item,
        returns one result.
    max_workers
        Number of worker processes.
    max_pending
        Maximum in-flight futures. Defaults to ``4 * max_workers``.
    timeout
        Timeout in seconds. In ordered mode this is applied while waiting for
        each submitted future. In unordered mode it is the maximum wait for the
        current pending batch to produce the next completed future, matching
        ``concurrent.futures.as_completed`` semantics. Default 300s. Set to
        0 to disable.
    ordered
        If True, yield results in submission order (preserves input order).
        If False (default), yield in completion order (faster throughput).
    initializer
        Callable run once per worker process at startup. Used to set up
        non-picklable state (e.g. standardizer objects) in a module global.
    initargs
        Arguments for ``initializer``.
    max_tasks_per_child
        Recycle worker processes after this many tasks (Python 3.11+).
        Prevents memory leaks in long-running pipelines.
    on_timeout
        Callback ``(timeout_error, original_item) -> fallback_result``.
        If provided, timeout does not raise — the callback's return value
        is yielded instead. If None, TimeoutError propagates.

    Yields
    ------
    Results from ``worker_fn`` in the order determined by ``ordered``.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    max_pending = max_pending or (4 * max_workers)
    if max_pending < 1:
        max_pending = 1

    effective_timeout = timeout if timeout > 0 else None

    if _RECYCLE_WORKERS and max_tasks_per_child is not None:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
            max_tasks_per_child=max_tasks_per_child,
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
        )
    completed = False
    try:
        if ordered:
            yield from _ordered_stream(
                executor,
                items,
                worker_fn,
                max_pending,
                effective_timeout,
                on_timeout,
            )
        else:
            yield from _unordered_stream(
                executor,
                items,
                worker_fn,
                max_pending,
                effective_timeout,
                on_timeout,
            )
        completed = True
    finally:
        force_terminate = not completed or getattr(
            executor, _FORCE_TERMINATE_ATTR, False
        )
        _shutdown_process_pool_executor(executor, force_terminate=force_terminate)


def _request_executor_termination(executor: ProcessPoolExecutor) -> None:
    setattr(executor, _FORCE_TERMINATE_ATTR, True)


def _shutdown_process_pool_executor(
    executor: ProcessPoolExecutor,
    *,
    force_terminate: bool,
) -> None:
    processes = _executor_processes(executor)
    manager_thread = getattr(executor, "_executor_manager_thread", None)

    if force_terminate and _terminate_executor_publicly(executor):
        _join_manager_thread(manager_thread)
        return

    # Do not use shutdown(wait=True) directly: some chemistry workers finish
    # their tasks but keep non-daemon native/runtime threads alive, which makes
    # the executor join hang and leaves stale processes after CLI completion.
    _shutdown_executor_without_wait(executor)

    if force_terminate:
        _terminate_processes(processes)
    else:
        _join_processes(processes, timeout=_PROCESS_EXIT_TIMEOUT)
        _terminate_processes(_live_processes(processes))

    _join_manager_thread(manager_thread)


def _terminate_executor_publicly(executor: ProcessPoolExecutor) -> bool:
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is None:
        return False
    # Python 3.14 added the public API we want here. Calixarene currently runs
    # Python 3.13, so older interpreters fall back to cached private processes.
    with contextlib.suppress(Exception):
        terminate_workers()
        return True
    return False


def _shutdown_executor_without_wait(executor: ProcessPoolExecutor) -> None:
    if _CANCEL_FUTURES:
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=False)


def _executor_processes(executor: ProcessPoolExecutor) -> list[Any]:
    processes = getattr(executor, "_processes", None)
    if not processes:
        return []
    return [process for process in processes.values() if process is not None]


def _terminate_processes(
    processes: list[Any],
    *,
    terminate_timeout: float = _TERMINATE_TIMEOUT,
    kill_timeout: float = _KILL_TIMEOUT,
) -> None:
    if not processes:
        return

    live_processes = _live_processes(processes)
    for process in live_processes:
        with contextlib.suppress(Exception):
            process.terminate()

    _join_processes(live_processes, timeout=terminate_timeout)

    stubborn_processes = _live_processes(live_processes)
    for process in stubborn_processes:
        kill = getattr(process, "kill", None)
        if kill is not None:
            with contextlib.suppress(Exception):
                kill()

    _join_processes(stubborn_processes, timeout=kill_timeout)


def _join_processes(processes: list[Any], *, timeout: float) -> None:
    for process in processes:
        with contextlib.suppress(Exception):
            process.join(timeout=timeout)


def _live_processes(processes: list[Any]) -> list[Any]:
    return [process for process in processes if _process_is_alive(process)]


def _join_manager_thread(thread: Any) -> None:
    if thread is None:
        return
    with contextlib.suppress(Exception):
        thread.join(timeout=_MANAGER_THREAD_TIMEOUT)


def _process_is_alive(process: Any) -> bool:
    is_alive = getattr(process, "is_alive", None)
    if is_alive is None:
        return False
    with contextlib.suppress(Exception):
        return bool(is_alive())
    return False


def _ordered_stream(
    executor: ProcessPoolExecutor,
    items: Iterable[Any],
    worker_fn: Callable,
    max_pending: int,
    timeout: float | None,
    on_timeout: Callable | None,
) -> Iterator[Any]:
    """Yield results in submission order using a deque."""
    futures_deque: deque[tuple[Future, Any]] = deque()
    iterator = iter(items)
    exhausted = False

    def _fill():
        nonlocal exhausted
        while len(futures_deque) < max_pending and not exhausted:
            try:
                item = next(iterator)
                fut = executor.submit(worker_fn, item)
                futures_deque.append((fut, item))
            except StopIteration:
                exhausted = True
                break

    _fill()

    while futures_deque:
        fut, original_item = futures_deque[0]
        try:
            result = fut.result(timeout=timeout)
            futures_deque.popleft()
            _fill()
            yield result
        except FuturesTimeoutError as e:
            _request_executor_termination(executor)
            futures_deque.popleft()
            _fill()
            if on_timeout is not None:
                yield on_timeout(e, original_item)
            else:
                raise


def _unordered_stream(
    executor: ProcessPoolExecutor,
    items: Iterable[Any],
    worker_fn: Callable,
    max_pending: int,
    timeout: float | None,
    on_timeout: Callable | None,
) -> Iterator[Any]:
    """Yield results in completion order.

    ``as_completed(..., timeout=...)`` applies one deadline to the current
    pending set, not an independent deadline per future. This is intentional
    here because unordered mode optimizes throughput and only needs to detect
    a fully stalled pending window.
    """
    iterator = iter(items)
    pending: dict[Future, Any] = {}

    # Prime
    try:
        while len(pending) < max_pending:
            item = next(iterator)
            fut = executor.submit(worker_fn, item)
            pending[fut] = item
    except StopIteration:
        pass

    while pending:
        try:
            done_futures = as_completed(pending, timeout=timeout)
            for future in done_futures:
                original_item = pending.pop(future)

                # Refill
                try:
                    while len(pending) < max_pending:
                        item = next(iterator)
                        fut = executor.submit(worker_fn, item)
                        pending[fut] = item
                except StopIteration:
                    pass

                try:
                    yield future.result()
                except FuturesTimeoutError as e:
                    _request_executor_termination(executor)
                    if on_timeout is not None:
                        yield on_timeout(e, original_item)
                    else:
                        raise
        except FuturesTimeoutError:
            _request_executor_termination(executor)
            raise


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


@contextmanager
def graceful_shutdown() -> Iterator[threading.Event]:
    """Context manager that catches SIGTERM/SIGINT and sets a stop event.

    Restores the original signal handlers on exit. If called from a non-main
    thread, signal handling is silently skipped (the stop event is still
    usable for manual signaling).

    Usage::

        with graceful_shutdown() as stop:
            for chunk in process_pool_map_stream(...):
                if stop.is_set():
                    break
                process(chunk)
    """
    stop = threading.Event()
    originals: dict[int, Any] = {}

    def _handler(*_: Any) -> None:
        logger.warning("Received termination signal — stopping gracefully")
        stop.set()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            originals[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
    except ValueError:
        # Not on main thread — signal handling unavailable.
        pass

    try:
        yield stop
    finally:
        for sig, original in originals.items():
            with contextlib.suppress(ValueError):
                signal.signal(sig, original)


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def chunked(iterable: Iterable, size: int) -> Iterator[list]:
    """Yield successive chunks of *size* items from *iterable*."""
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
