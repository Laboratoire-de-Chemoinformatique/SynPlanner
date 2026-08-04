"""Metadata helpers shared by route JSON and CSV codecs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


def reaction_metadata(reaction) -> dict[str, Any]:
    metadata = getattr(reaction, "meta", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def restore_reaction_metadata(reaction, metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        return
    existing = getattr(reaction, "meta", None)
    if isinstance(existing, dict):
        existing.update(metadata)
        return
    try:
        reaction.meta = dict(metadata)
    except (AttributeError, TypeError):
        logger.warning("Route reaction metadata could not be attached")
