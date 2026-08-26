"""Shrinks a page full of chython depictions.

Whitespace and numeric literals are tightened losslessly, and identical molecules are
defined once in a shared pool that every occurrence reaches through ``<use>``.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["Doc", "content_key", "depictions", "hidden_defs", "tighten"]

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_WS = re.compile(r">\s+<")
_ATTR = re.compile(r'="([^"]*)"')
_NUM = re.compile(r"-?\d+\.\d+")
_B36 = "abcdefghijklmnopqrstuvwxyz0123456789"


def _b36(n: int) -> str:
    out = ""
    while True:
        n, r = divmod(n, 36)
        out = _B36[r] + out
        if not n:
            return out


def _shortest(literal: str) -> str:
    """Shortest literal parsing to the same double. Not rounding — the same number."""
    short = repr(float(literal))
    if short.endswith(".0"):
        short = short[:-2]
    if short.startswith("0."):
        short = short[1:]
    elif short.startswith("-0."):
        short = "-" + short[2:]
    if len(short) < len(literal) and float(short) == float(literal):
        return short
    return literal


def tighten(svg: str) -> str:
    """Drop chython's pretty-print and its padding zeros. No rendering effect."""
    svg = _WS.sub("><", svg).strip()
    return _ATTR.sub(
        lambda m: '="' + _NUM.sub(lambda n: _shortest(n.group(0)), m.group(1)) + '"',
        svg,
    )


def depictions(svg: str) -> list[tuple[int, int, str, str]]:
    """``(start, end, opening tag, body)`` per nested depiction, in document order.

    A chython depiction holds no nested ``<svg>``, so the first ``</svg>`` after the
    wrapper closes it.
    """
    out, i = [], 0
    while True:
        i = svg.find('<svg x="', i)
        if i < 0:
            return out
        head = svg.index(">", i) + 1
        end = svg.index("</svg>", head)
        out.append((i, end + 6, svg[i:head], svg[head:end]))
        i = end + 6


def content_key(body: str) -> str:
    """Content hash of a depiction with its per-render UUID normalised away."""
    return hashlib.sha1(_UUID.sub("U", tighten(body)).encode()).hexdigest()


class Doc:
    """Many routes, one pool of molecule definitions."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._defs: list[str] = []
        self.hits = 0

    def route(self, svg: str) -> str:
        """Replace every depiction in ``svg`` with a ``<use>`` of the pooled copy."""
        out, last = [], 0
        for start, end, head, body in depictions(svg):
            key = content_key(body)
            self.hits += 1
            pool_id = self._ids.get(key)
            if pool_id is None:
                pool_id = self._ids[key] = _b36(len(self._ids))
                self._defs.append(_UUID.sub(pool_id, tighten(body)))
            out.append(svg[last:start])
            out.append(f'{head}<use xlink:href="#{pool_id}-molecule"/></svg>')
            last = end
        out.append(svg[last:])
        return tighten("".join(out))

    def defs(self) -> str:
        return "".join(self._defs)

    def __len__(self) -> int:
        return len(self._ids)


def hidden_defs(*chunks: str) -> str:
    """An off-canvas ``<svg>`` holding the document's shared definitions."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" width="0" height="0" '
        f'style="position:absolute"><defs>{tighten("".join(chunks))}</defs></svg>'
    )
