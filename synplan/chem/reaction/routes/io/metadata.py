"""Metadata helpers shared by route JSON and CSV codecs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reaction_metadata(reaction) -> dict[str, Any]:
    """The reaction's metadata as a plain dict, detached from the reaction."""

    return dict(reaction.meta)


def restore_reaction_metadata(reaction, metadata: Any) -> None:
    """Put a file's metadata back on a reaction.

    ``chython`` exposes ``meta`` as a lazily built dict with no setter, so this
    updates rather than assigns. Anything the file wrote that is not a mapping is
    ignored -- the metadata block is the one part of a route node with no shape.
    """

    if isinstance(metadata, Mapping):
        reaction.meta.update(metadata)
