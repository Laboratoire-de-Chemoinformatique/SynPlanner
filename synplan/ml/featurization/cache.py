"""Shared bounded LRU cache for tensorized rule representations."""

from __future__ import annotations

from collections import OrderedDict

import torch

#: Default maximum number of entries retained by a rule fingerprint cache.
MAX_RULE_FINGERPRINT_CACHE_SIZE = 8


def cache_get(cache: OrderedDict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    """Return the cached tensor for ``key`` and mark it most-recently used."""
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def cache_set(
    cache: OrderedDict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
    *,
    max_size: int = MAX_RULE_FINGERPRINT_CACHE_SIZE,
) -> None:
    """Insert ``value`` for ``key`` and evict the oldest entries past ``max_size``."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


__all__ = [
    "MAX_RULE_FINGERPRINT_CACHE_SIZE",
    "cache_get",
    "cache_set",
]
