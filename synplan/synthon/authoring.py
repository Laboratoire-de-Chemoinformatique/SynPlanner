"""The check a ring-forming rule must pass before it is added to ``rules.json``.

A ring synthon carries two labels, and chython standardises tautomers on canonicalisation. Where a
labelled atom sits in an amide, amidine, thioamide or enol triad, the proton moves and the label
stays put, so the fragment the rule wrote is not the fragment the pipeline gets. The two nitrogens
of an amidine are genuinely equivalent — RDKit collapses them too — so this cannot be repaired
downstream. It has to be refused at authoring time.
"""

from chython.containers import SynthonContainer

from synplan.chem.utils import safe_canonicalization

__all__ = ["labelled_atoms_survive_canonicalisation", "shifted_labels"]


def _labelled(synthon: SynthonContainer) -> dict[int, tuple[str, int, str]]:
    """Element, implicit hydrogens and token of every labelled atom, keyed by atom number."""
    return {
        n: (atom.atomic_symbol, atom.implicit_hydrogens, atom.label)
        for n, atom in synthon.atoms()
        if getattr(atom, "_label", None) is not None
    }


def shifted_labels(synthon: SynthonContainer) -> dict[int, tuple]:
    """Labelled atoms whose hydrogen count canonicalisation changes, written form vs canonical.

    :param synthon: A synthon exactly as the rule produced it, **not** yet canonicalised.
    :return: ``{atom: (before, after)}``, empty when the written form survives.
    """
    before = _labelled(synthon)
    after = _labelled(safe_canonicalization(synthon.copy()))
    return {n: (before[n], after[n]) for n in before if before.get(n) != after.get(n)}


def labelled_atoms_survive_canonicalisation(synthon: SynthonContainer) -> bool:
    """Whether the regiochemistry the rule spells is the one the pipeline will see."""
    return not shifted_labels(synthon)
