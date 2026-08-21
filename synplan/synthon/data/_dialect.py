"""RDKit-SMARTS to chython-SMARTS shim. Build-time only — the shipped JSON is already translated.

Five mechanical rewrites, no literal overrides, applied by a recursive walk so that bracket bodies,
recursive ``$()`` sub-patterns and bare atoms are each handled in their own context. Anything
outside this corpus's idioms must raise rather than degrade, so a pattern this cannot express shows
up in the converter's exit code instead of silently matching the wrong thing.
"""

import re

from chython.periodictable import Element

# `!c` in primitive position: chython's _is_element_token admits only capitalised symbols, so a
# negated aromatic symbol never reaches the branch that would handle it.
_AROMATIC = ("c", "n", "o", "p", "s", "b", "se", "te", "as")

# rewrite 1: Synt-On writes `!@-`; chython finalises a bare `!@` into a bond token, so the
# trailing order becomes a second bond. The order has to come first.
_BOND_RING_FLAG = re.compile(r"(!?@)([-=:#~])")
# rewrite 3: a bare organic-subset atom is not aliphatic in chython — `C[Cl,Br,I]` matches an
# *aryl* iodide. Only bracketed symbols get hybridization [1, 2, 3].
_ORGANIC_SUBSET = re.compile(r"Cl|Br|[BCNOPSFI]")
# a `,`-OR member that is a plain element symbol or atomic number; anything else is a compound
# primitive that chython refuses to OR with a different kind.
_BARE_ELEMENT = re.compile(r"^#[0-9]+$|^[A-Za-z][a-z]?$")
_ATOM_MAP = re.compile(r":[1-9][0-9]*$")
_BOND_OR_RING = re.compile(r"[-=:#~]!?@(?:,[-=:#~]!?@)+")


class DialectError(ValueError):
    """A pattern this shim cannot express in the chython dialect."""


def _match(text: str, i: int, closing: str) -> int:
    """Index of the bracket/paren closing the one that opens at *i*."""
    opening = text[i]
    depth = 0
    for j in range(i, len(text)):
        if text[j] == opening:
            depth += 1
        elif text[j] == closing:
            depth -= 1
            if not depth:
                return j
    raise DialectError(f"unbalanced {opening} in {text!r}")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on *sep* outside any bracket or paren nesting."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "[(":
            depth += 1
        elif c in "])":
            depth -= 1
        elif c == sep and not depth:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def _hoist_ring_flag(match: re.Match) -> str:
    """rewrite 5: chython cannot OR ring-flagged bonds (`=@,:@`); the flag hoists with `;`."""
    members = match.group().split(",")
    flags = {m[1:] for m in members}
    if len(flags) != 1:
        raise DialectError(f"mixed ring flags in bond OR {match.group()!r}")
    return f"{','.join(m[0] for m in members)};{flags.pop()}"


def _translate_outside(run: str) -> str:
    """Bond and bare-atom context: swap the ring flag, then bracket organic-subset atoms."""
    run = _BOND_RING_FLAG.sub(r"\2\1", run)
    run = _BOND_OR_RING.sub(_hoist_ring_flag, run)
    return _ORGANIC_SUBSET.sub(lambda m: f"[{m.group()}]", run)


def _wrap_or(chunk: str) -> str:
    """Rewrite 2: an OR of non-element primitives becomes an OR of recursive primitives."""
    members = _split_top_level(chunk, ",")
    if len(members) == 1 or all(_BARE_ELEMENT.match(m) for m in members):
        return chunk
    return ",".join(
        m if m.lstrip("!").startswith("$") else f"$([{m}])" for m in members
    )


def _translate_bracket(body: str) -> str:
    """Bracket body: descend into `$()`, AND juxtaposed recursives, then fix the ORs."""
    # the atom map belongs to the whole atom, so it must not be swept into a `,`-member's `$()`
    map_match = _ATOM_MAP.search(body)
    if map_match:
        return _translate_bracket(body[: map_match.start()]) + map_match.group()
    out, i, previous_was_recursive = [], 0, False
    while i < len(body):
        if body[i] == "$" or (body[i] == "!" and body[i + 1 : i + 2] == "$"):
            negated = body[i] == "!"
            open_paren = i + 2 if negated else i + 1
            if body[open_paren : open_paren + 1] != "(":
                raise DialectError(f"malformed recursive primitive in [{body}]")
            close = _match(body, open_paren, ")")
            if previous_was_recursive:
                # rewrite 4: Daylight juxtaposition is AND; chython ORs same-term constraints
                out.append(";")
            out.append(
                f"{'!' if negated else ''}$({_translate(body[open_paren + 1 : close])})"
            )
            i = close + 1
            previous_was_recursive = True
        else:
            out.append(body[i])
            previous_was_recursive = False
            i += 1
    chunks = _split_top_level("".join(out), ";")
    return ";".join(_wrap_or(_negated_aromatic(c, chunks)) for c in chunks)


def _negated_aromatic(chunk: str, chunks: list[str]) -> str:
    """`[a;!c]` -> `[a;!#6]`. Only equivalent because the bracket already asserts aromaticity."""
    if not (chunk.startswith("!") and chunk[1:] in _AROMATIC):
        return chunk
    if "a" not in chunks:
        raise DialectError(
            f"!{chunk[1:]} outside an aromatic bracket has no chython spelling"
        )
    return f"!#{Element.from_symbol(chunk[1:].capitalize())().atomic_number}"


def _translate(pattern: str) -> str:
    out, i = [], 0
    while i < len(pattern):
        if pattern[i] == "[":
            close = _match(pattern, i, "]")
            out.append(f"[{_translate_bracket(pattern[i + 1 : close])}]")
            i = close + 1
        else:
            j = pattern.find("[", i)
            j = len(pattern) if j < 0 else j
            out.append(_translate_outside(pattern[i:j]))
            i = j
    return "".join(out)


def to_chython(pattern: str) -> str:
    """Translate one RDKit SMARTS into the chython dialect."""
    return _translate(pattern)


__all__ = ["DialectError", "to_chython"]
