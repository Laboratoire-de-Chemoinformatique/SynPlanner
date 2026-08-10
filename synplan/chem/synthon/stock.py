"""Synthon stock: text-level I/O, the two lookup indexes, Ro2 and leaving-group capping."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property

from chython import synthon_smiles
from chython.containers import SynthonContainer
from chython.periodictable.base.synthon import BIVALENT_LABELS
from rdkit.Chem import AddHs, Crippen
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBA, CalcNumHBD

from synplan.chem.synthon.analogues import (
    analogue_key,
    find_analogues,
    index_for_analogues,
)
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.utils.files import iter_smiles_records

# MW > 200 or logP > 2 or HBD > 2 or HBA > 4 rejects. Identical across all three upstream
# implementations; only the descriptor corrections differ.
RO2_LIMITS = (200.0, 2.0, 2, 4)


@dataclass(frozen=True, slots=True)
class SynthonRecord:
    synthon: str
    building_blocks: tuple[str, ...]
    classes: tuple[str, ...]
    component: int

    def line(self) -> str:
        return (
            f"{self.synthon}\t{'+'.join(self.building_blocks)}\t"
            f"{'+'.join(self.classes)}\t{self.component}"
        )


def read_synthon_records(path: str) -> Iterator[SynthonRecord]:
    """Parse the stock file. Text-level: MoleculeReader hard-codes plain `smiles` and would raise."""
    for line in iter_smiles_records(path):
        fields = line.split("\t")
        if len(fields) < 4:
            fields = line.split()
        synthon, blocks, classes, component = fields[:4]
        yield SynthonRecord(
            synthon, tuple(blocks.split("+")), tuple(classes.split("+")), int(component)
        )


class SynthonStock(dict):
    """`synthon SMILES -> the building blocks that produce it`, plus the slot lookup.

    Still a dict, so `s in stock` and `Fragmenter(config, stock)` are unchanged.
    """

    @cached_property
    def _analogue_index(self) -> dict[tuple, list[str]]:
        return index_for_analogues(self)

    def slots(
        self, synthons: Iterable[str], config: SynthonConfig | None = None
    ) -> dict[str, list[str]]:
        """What may fill each slot of a fragmentation pathway.

        The stocked synthon itself, plus its positional analogues when `find_analogues`, minus
        whatever Ro2 rejects. An empty slot is a real answer — `strict_availability` decides
        whether it kills the pathway.
        """
        config = config or SynthonConfig()
        out: dict[str, list[str]] = {}
        for smi in synthons:
            found = [smi] if smi in self else []
            if config.find_analogues:
                found += [
                    other
                    for other in find_analogues(
                        synthon_smiles(smi),
                        self._analogue_index,
                        config.similarity_threshold,
                        config.pas_removal_direction,
                    )
                    if other not in found
                ]
            out[smi] = ro2_filter(found, config)
        return out


def _ro2_or_reject(smi: str, variant: str) -> bool:
    """A synthon RDKit refuses to sanitise cannot be shown to be reagent-like, so it is not.

    Two rows in 6424 from a real catalogue carry a double-bonded `[O-]`; one of them must not take
    the whole stock load down with it.
    """
    try:
        return ro2_pass(synthon_smiles(smi), variant)
    except Exception:
        return False


def ro2_filter(
    synthons: Iterable[str], config: SynthonConfig | None = None
) -> list[str]:
    """Drop the synthons the rule of two rejects. `ro2_filtration` off is the identity."""
    config = config or SynthonConfig()
    if not config.ro2_filtration:
        return list(synthons)
    return [s for s in synthons if _ro2_or_reject(s, config.ro2_variant)]


def load_synthon_stock(path: str, config: SynthonConfig | None = None) -> SynthonStock:
    """synthon SMILES -> the building blocks that produce it, Ro2-filtered when configured."""
    stock: dict[str, set[str]] = {}
    for record in read_synthon_records(path):
        stock.setdefault(record.synthon, set()).update(record.building_blocks)
    keep = set(ro2_filter(stock, config))
    return SynthonStock((s, b) for s, b in stock.items() if s in keep)


def write_synthon_stock(path: str, records: Iterable[SynthonRecord]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.line() + "\n")


def label_keys(synthon: SynthonContainer) -> list[tuple[str, bool, str]]:
    """(symbol, aromatic, token) per labelled atom. Aromaticity is `hybridization == 4`, never
    the case of the symbol."""
    return [
        (atom.atomic_symbol, atom.hybridization == 4, atom.label)
        for _, atom in synthon.atoms()
        if getattr(atom, "_label", None) is not None
    ]


def index_by_label(stock: Iterable[str]) -> dict[tuple[str, bool, str], list[str]]:
    """(symbol, aromatic, token) -> stocked synthons carrying it. The forward action space."""
    index: dict[tuple[str, bool, str], list[str]] = {}
    for smi in stock:
        for key in set(label_keys(synthon_smiles(smi))):
            index.setdefault(key, []).append(smi)
    return index


def ro2_pass(synthon: SynthonContainer, variant: str = "paper") -> bool:
    """Rule of two. `paper` reproduces the published numbers; `corrected` is label-aware.

    The published Fig. 5 numbers come from the implementation that does NOT apply the reference's
    own documented corrections — the README documents a helper it does not call.
    """
    plain = synthon.unlabelled().to_rdkit(keep_mapping=False)
    with_hydrogens = AddHs(plain)
    mass = CalcExactMolWt(with_hydrogens)
    donors = CalcNumHBD(with_hydrogens)
    acceptors = CalcNumHBA(with_hydrogens)
    logp = Crippen.MolLogP(plain)
    if variant == "corrected":
        points = sum(
            2 if a.label in BIVALENT_LABELS else 1
            for _, a in synthon.atoms()
            if getattr(a, "_label", None) is not None
        )
        mass -= 1.00783 * points
        donors -= sum(
            1
            for _, a in synthon.atoms()
            if getattr(a, "_label", None) is not None
            and a.atomic_symbol in ("N", "O", "S")
            and a.implicit_hydrogens
        )
    limit_mass, limit_logp, limit_donors, limit_acceptors = RO2_LIMITS
    return not (
        mass > limit_mass
        or logp > limit_logp
        or donors > limit_donors
        or acceptors > limit_acceptors
    )


def cap_leaving_group(
    symbol: str, aromatic: bool, token: str, config: SynthonConfig | None = None
) -> str:
    """One representative leaving group per attachment point, so a planner can name a reagent."""
    # ponytail: two entries are shorthand, not reagents — `C:nuc`/`c:nuc` give `[Mg]` where a
    # Grignard is `[Mg]Br`, and `C:nuc*`/`c:nuc*` give a trifluoroborate with no counter-cation.
    # Zero call sites today; upgrade path is to spell the whole reagent in `rules.json`.
    rules = load_data((config or SynthonConfig()).rules_path)["leaving_groups"]
    key = f"{symbol.lower() if aromatic else symbol}:{token}"
    return rules.get(key, "H")


__all__ = [
    "RO2_LIMITS",
    "SynthonRecord",
    "SynthonStock",
    "analogue_key",
    "cap_leaving_group",
    "index_by_label",
    "index_for_analogues",
    "label_keys",
    "load_synthon_stock",
    "read_synthon_records",
    "ro2_filter",
    "ro2_pass",
    "write_synthon_stock",
]
