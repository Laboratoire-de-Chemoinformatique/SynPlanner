"""Generic parallelization utilities.

This module provides small, reusable helpers for ProcessPool-based
parallel mapping with backpressure. Keep format-specific I/O utilities in
`synplan.utils.files`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import torch


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
) -> Iterator[Any]:
    """Submit tasks lazily and yield results as they finish.

    Limits the number of in‑flight futures to avoid memory spikes when `items`
    is a large or infinite iterator. Results are yielded in completion order.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    max_pending = max_pending or (4 * max_workers)
    if max_pending < 1:
        max_pending = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        iterator = iter(items)
        pending = set()

        # Prime the queue up to max_pending
        try:
            while len(pending) < max_pending:
                pending.add(executor.submit(worker_fn, next(iterator)))
        except StopIteration:
            pass

        while pending:
            for future in as_completed(pending):
                pending.remove(future)

                # Refill the queue immediately to keep workers busy
                try:
                    while len(pending) < max_pending:
                        pending.add(executor.submit(worker_fn, next(iterator)))
                except StopIteration:
                    pass

                yield future.result()
