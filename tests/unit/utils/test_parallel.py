"""Tests for process_pool_map_stream and graceful_shutdown.

Covers: ordered mode, unordered mode, timeout, initializer pattern,
backpressure, and graceful shutdown.
"""

import signal
import sys
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

import synplan.utils.parallel as parallel
from synplan.utils.parallel import (
    graceful_shutdown,
    process_pool_map_stream,
)

# -- Test helpers (module-level for pickling) --------------------------------

_worker_multiplier = None


def _init_multiply(factor: int):
    global _worker_multiplier
    _worker_multiplier = factor


def _multiply(x: int) -> int:
    if _worker_multiplier is None:
        raise RuntimeError("Worker not initialized")
    return x * _worker_multiplier


def _identity(x):
    return x


def _slow_identity(x):
    time.sleep(0.5)
    return x


def _hang_on_value(x):
    """Hang forever if x == -1, otherwise return x."""
    if x == -1:
        time.sleep(9999)
    return x


# -- T1: Ordered mode preserves submission order -----------------------------


def test_ordered_preserves_order():
    """Items with variable processing time still come out in submission order."""
    items = list(range(20))
    results = list(
        process_pool_map_stream(
            items,
            _identity,
            max_workers=4,
            ordered=True,
            timeout=10,
        )
    )
    assert results == items


# -- T2: Unordered mode yields all results -----------------------------------


def test_unordered_yields_all():
    """All items are yielded (order may differ)."""
    items = list(range(20))
    results = list(
        process_pool_map_stream(
            items,
            _identity,
            max_workers=4,
            ordered=False,
            timeout=10,
        )
    )
    assert sorted(results) == items


# -- T3: Timeout yields error via on_timeout ---------------------------------


def test_timeout_with_callback():
    """When a future times out, on_timeout callback is invoked."""
    items = [1, 2, -1, 4]  # -1 hangs forever

    timed_out = []

    def on_timeout(exc, original_item):
        timed_out.append(original_item)
        return f"timeout:{original_item}"

    results = list(
        process_pool_map_stream(
            items,
            _hang_on_value,
            max_workers=2,
            ordered=True,
            timeout=5,  # generous timeout for slow CI (Windows)
            on_timeout=on_timeout,
        )
    )

    assert 1 in results
    assert 2 in results
    assert 4 in results
    assert "timeout:-1" in results
    assert -1 in timed_out


# -- T4: Timeout without callback raises ------------------------------------


def test_timeout_raises_without_callback():
    """Without on_timeout, TimeoutError propagates."""
    from concurrent.futures import TimeoutError as FTE

    items = [-1]  # hangs

    with pytest.raises(FTE):
        list(
            process_pool_map_stream(
                items,
                _hang_on_value,
                max_workers=1,
                ordered=True,
                timeout=1,
            )
        )


# -- T5: Initializer pattern works ------------------------------------------


def test_initializer_sets_worker_state():
    """Worker state set by initializer is available in worker function."""
    items = [1, 2, 3, 4, 5]
    results = list(
        process_pool_map_stream(
            items,
            _multiply,
            max_workers=2,
            ordered=True,
            timeout=10,
            initializer=_init_multiply,
            initargs=(10,),
        )
    )
    assert results == [10, 20, 30, 40, 50]


# -- T6: Backpressure limits in-flight futures -------------------------------


def test_backpressure():
    """max_pending limits concurrent futures."""
    # With slow workers and small max_pending, we shouldn't blow up memory
    items = list(range(10))
    results = list(
        process_pool_map_stream(
            items,
            _slow_identity,
            max_workers=2,
            max_pending=3,
            ordered=True,
            timeout=30,
        )
    )
    assert results == items


# -- T7: Empty input ---------------------------------------------------------


def test_empty_input():
    """Empty iterable yields no results."""
    results = list(
        process_pool_map_stream(
            [],
            _identity,
            max_workers=1,
            timeout=10,
        )
    )
    assert results == []


# -- T8: Single worker (serial) ---------------------------------------------


def test_single_worker():
    """max_workers=1 works correctly (serial processing)."""
    items = list(range(5))
    results = list(
        process_pool_map_stream(
            items,
            _identity,
            max_workers=1,
            ordered=True,
            timeout=10,
        )
    )
    assert results == items


def test_process_pool_shutdown_joins_on_normal_completion(monkeypatch):
    """Normal completion joins worker processes instead of leaving cleanup to exit."""

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.join_calls = []
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminated = True
            self._alive = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            self._alive = False

    class FakeExecutor:
        instances = []

        def __init__(self, *args, **kwargs):
            self.process = FakeProcess()
            self._processes = {1: self.process}
            self.shutdown_calls = []
            FakeExecutor.instances.append(self)

        def submit(self, fn, item):
            return ImmediateFuture(fn(item))

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)

    assert list(
        process_pool_map_stream([1], _identity, max_workers=1, ordered=True)
    ) == [1]

    executor = FakeExecutor.instances[0]
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert executor.process.join_calls
    assert executor.process.terminated is False


def test_process_pool_terminates_stale_workers_after_completion(monkeypatch):
    """Finished tasks may leave non-exiting workers; cleanup must kill those."""

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    class StaleProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.join_calls = []

        def is_alive(self):
            return not self.terminated and not self.killed

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakeExecutor:
        instances = []

        def __init__(self, *args, **kwargs):
            self.process = StaleProcess()
            self._processes = {1: self.process}
            self.shutdown_calls = []
            FakeExecutor.instances.append(self)

        def submit(self, fn, item):
            return ImmediateFuture(fn(item))

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)

    assert list(
        process_pool_map_stream([1], _identity, max_workers=1, ordered=True)
    ) == [1]

    executor = FakeExecutor.instances[0]
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert executor.process.terminated is True


def test_process_pool_terminates_workers_after_timeout_callback(monkeypatch):
    """A timed-out future with fallback must still terminate stale workers."""

    class TimeoutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError()

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.join_calls = []

        def is_alive(self):
            return not self.terminated and not self.killed

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakeExecutor:
        instances = []

        def __init__(self, *args, **kwargs):
            self.process = FakeProcess()
            self._processes = {1: self.process}
            self.shutdown_calls = []
            FakeExecutor.instances.append(self)

        def submit(self, fn, item):
            return TimeoutFuture()

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)

    results = list(
        process_pool_map_stream(
            [1],
            _identity,
            max_workers=1,
            ordered=True,
            timeout=0.01,
            on_timeout=lambda _exc, item: f"timeout:{item}",
        )
    )

    executor = FakeExecutor.instances[0]
    assert results == ["timeout:1"]
    assert executor.process.terminated is True
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_process_pool_uses_public_terminate_workers_when_available(monkeypatch):
    """Python 3.14+ exposes public worker termination; prefer it when present."""

    class TimeoutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError()

    class FakeExecutor:
        instances = []

        def __init__(self, *args, **kwargs):
            self.shutdown_calls = []
            self.terminate_workers_called = False
            FakeExecutor.instances.append(self)

        def submit(self, fn, item):
            return TimeoutFuture()

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

        def terminate_workers(self):
            self.terminate_workers_called = True

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)

    results = list(
        process_pool_map_stream(
            [1],
            _identity,
            max_workers=1,
            ordered=True,
            timeout=0.01,
            on_timeout=lambda _exc, item: f"timeout:{item}",
        )
    )

    executor = FakeExecutor.instances[0]
    assert results == ["timeout:1"]
    assert executor.terminate_workers_called is True
    assert executor.shutdown_calls == []


# -- T9: graceful_shutdown sets stop event on signal -------------------------

_is_windows = sys.platform == "win32"


@pytest.mark.skipif(
    _is_windows,
    reason="os.kill(SIGTERM) terminates the process on Windows; handler cannot catch it",
)
def test_graceful_shutdown_sigterm():
    """SIGTERM sets the stop event (POSIX only)."""
    with graceful_shutdown() as stop:
        assert not stop.is_set()
        import os

        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.1)
        assert stop.is_set()


# -- T10: graceful_shutdown restores original handler -----------------------


@pytest.mark.skipif(
    _is_windows,
    reason="os.kill(SIGTERM) terminates the process on Windows; handler cannot catch it",
)
def test_graceful_shutdown_restores_handler():
    """Original signal handler is restored after exit (POSIX only)."""
    original = signal.getsignal(signal.SIGTERM)
    with graceful_shutdown() as _:
        assert signal.getsignal(signal.SIGTERM) != original
    assert signal.getsignal(signal.SIGTERM) is original
