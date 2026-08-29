"""Bemis-Murcko scaffolds after removing the ring-containing protecting/leaving groups."""

from chython import smiles
from chython.containers import MoleculeContainer
from chython.files import smarts
from chython.reactor import Transformer

from synplan.chem.utils import safe_canonicalization

# six PG/LG removals applied to fixpoint, so a di-Cbz piperazine loses both. Each deletes the
# WHOLE group and caps with H; Boc is absent because plain Murcko already removes it.
PG_RULES = (
    "[N:1][C;$([C](=[O])[O][CH2]c1[cH][cH][cH][cH][cH]1):2]>>[N:1]",
    "[N:1][C;$([C](=[O])[O][CH2][CH]1c2[cH][cH][cH][cH]c2-c3[cH][cH][cH][cH]c13):2]>>[N:1]",
    "[N;+0;$([N][CH2]c1[cH][cH][cH][cH][cH]1);!$([N][C,S,P]=[O,S,N]):1]"
    "[C;$([CH2]c1[cH][cH][cH][cH][cH]1):2]>>[N:1]",
    "[O;$([O]([C])[C]([#6])=[O]):1][C;$([CH2]c1[cH][cH][cH][cH][cH]1):2]>>[OH:1]",
    # both oxygens go with the boron, so a pinacol boronate leaves the bare carbon behind
    "[B:1]([O:3])([O:4])[#6:2]>>[#6:2]",
    # not a deletion at all: an epoxide OPENING, which destroys the ring so Murcko cannot keep it
    "[C:1]1[O:2][C:3]1>>[C:1]([OH:2])[C;+0:3]",
)

LINEAR = "linearMolecule"


def _rules() -> list[tuple]:
    return [
        (
            smarts(r.split(">>")[0]),
            Transformer(smarts(r.split(">>")[0]), smarts(r.split(">>")[1])),
        )
        for r in PG_RULES
    ]


def strip_protecting_groups(molecule: MoleculeContainer) -> MoleculeContainer:
    """Apply the six rules to fixpoint."""
    for query, rule in _rules():
        while query.is_substructure(molecule):
            products = list(rule(molecule))
            if not products:
                break
            molecule = products[0]
    return molecule


def murcko_atoms(molecule: MoleculeContainer) -> set[int]:
    """Ring systems plus linkers, then exocyclic multiple bonds added back.

    The naive "delete degree-1 non-ring atoms to fixpoint" loop is WRONG: RDKit keeps atoms
    multiply bonded to scaffold atoms, such as the =O of an amide or sulfonamide linker.
    """
    keep = {n for ring in molecule.sssr for n in ring}
    if not keep:
        return set()
    core = set(molecule)
    while True:
        drop = {
            n
            for n in core
            if n not in keep and len(molecule._bonds[n].keys() & core) <= 1
        }
        if not drop:
            break
        core -= drop
    return core | {
        other
        for n in core
        for other, bond in molecule._bonds[n].items()
        if int(bond) > 1
    }


def murcko_scaffold(molecule: MoleculeContainer, strip: bool = True) -> str:
    """Bemis-Murcko scaffold SMILES, or `linearMolecule` when the input has no ring."""
    stripped = strip_protecting_groups(molecule) if strip else molecule
    core = murcko_atoms(stripped)
    if not core:
        return LINEAR
    scaffold = safe_canonicalization(stripped.substructure(core))
    return str(scaffold)


def scaffold_smiles(smi: str, strip: bool = True) -> str:
    return murcko_scaffold(safe_canonicalization(smiles(smi)), strip)


__all__ = [
    "LINEAR",
    "PG_RULES",
    "murcko_atoms",
    "murcko_scaffold",
    "scaffold_smiles",
    "strip_protecting_groups",
]
