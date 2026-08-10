"""Positional analogue scanning over the synthon stock."""

from collections import Counter
from collections.abc import Iterable

from chython import synthon_smiles
from chython.containers import SynthonContainer

# the four elements the reference lets an analogue gain or lose
_SWAPPABLE = frozenset({"C", "F", "N", "O"})


def analogue_key(synthon: SynthonContainer) -> tuple[tuple, tuple]:
    """The two hard PAS gates, precomputed: the label multiset and the degree signature.

    The degree signature is stricter than the paper's "same types of RCs": [NH2_nuc] (degree 1)
    and [NH_nuc] (degree 2) are not interchangeable.
    """
    labels = tuple(
        sorted(
            a.label
            for _, a in synthon.atoms()
            if getattr(a, "_label", None) is not None
        )
    )
    # heavy-neighbour degree, not total connectivity: [NH2_nuc] is degree 1 and [NH_nuc] is
    # degree 2, and they are not interchangeable
    signature = tuple(
        sorted(
            (a.atomic_symbol, len(synthon._bonds[n]))
            for n, a in synthon.atoms()
            if getattr(a, "_label", None) is not None
        )
    )
    return labels, signature


def index_for_analogues(stock: Iterable[str]) -> dict[tuple, list[str]]:
    """Both gates are exact equality, so they are a dict key, not a scan."""
    index: dict[tuple, list[str]] = {}
    for smi in stock:
        index.setdefault(analogue_key(synthon_smiles(smi)), []).append(smi)
    return index


def census(molecule: SynthonContainer) -> Counter:
    """Element counts on the GRAPH. Upstream scans the SMILES string, so `Cl` contributes C+l,
    `Br` contributes B+r, and a Cl->F change lands in no branch at all."""
    return Counter(atom.atomic_symbol for _, atom in molecule.atoms())


def tanimoto(a: SynthonContainer, b: SynthonContainer) -> float:
    left = a.morgan_bit_set(min_radius=1, max_radius=2, length=2048)
    right = b.morgan_bit_set(min_radius=1, max_radius=2, length=2048)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def is_analogue(
    reference: SynthonContainer,
    candidate: SynthonContainer,
    removal_direction: bool = True,
) -> bool:
    """Positional analogue scanning: same rings, at most one heavy atom apart, one of four shapes.

    The removal direction is `elif refList_qList and len(refList_qList) == 0` upstream — which is
    unsatisfiable for any list, so lines 80-88 never execute and an analogue may only ever GAIN a
    CH3/F/NH2/OH. The published rule is symmetric.
    """
    if len(reference.sssr) != len(candidate.sssr):
        return False
    if abs(len(reference) - len(candidate)) > 1:
        return False
    ref_census, cand_census = census(reference), census(candidate)
    if ref_census == cand_census:
        return True  # isomeric rearrangement
    gained = cand_census - ref_census
    lost = ref_census - cand_census
    if sum(gained.values()) == 1 and sum(lost.values()) == 1:
        # the aromatic C/N swap, decided on the graph rather than on lowercase letters
        return set(gained) | set(lost) == {"C", "N"}
    bare_reference, bare_candidate = reference.unlabelled(), candidate.unlabelled()
    if not lost and sum(gained.values()) == 1 and set(gained) <= _SWAPPABLE:
        return bare_reference.is_substructure(bare_candidate)
    if (
        removal_direction
        and not gained
        and sum(lost.values()) == 1
        and set(lost) <= _SWAPPABLE
    ):
        return bare_candidate.is_substructure(bare_reference)
    return False


def find_analogues(
    query: SynthonContainer,
    index: dict[tuple, list[str]],
    sim_threshold: float = -1.0,
    removal_direction: bool = True,
) -> list[str]:
    """Both gates first — they are exact equality, so O(1), which is what makes PAS tractable.

    The similarity branch is a union with PAS, not a replacement: `-1` disables it and leaves the
    PAS-only floor, and among thresholds >= 0 raising one strictly NARROWS the set back to it.
    """
    candidates = index.get(analogue_key(query), ())
    out = []
    for smi in candidates:
        candidate = synthon_smiles(smi)
        if str(candidate) == str(query):
            continue
        if (
            sim_threshold >= 0 and tanimoto(query, candidate) >= sim_threshold
        ) or is_analogue(query, candidate, removal_direction):
            out.append(smi)
    return out


__all__ = [
    "analogue_key",
    "census",
    "find_analogues",
    "index_for_analogues",
    "is_analogue",
    "tanimoto",
]
