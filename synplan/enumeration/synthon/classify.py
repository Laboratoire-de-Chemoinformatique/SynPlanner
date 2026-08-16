"""Building-block classification: 147 ordered classes over 2401 SMARTS."""

from chython import smarts, smiles
from chython.containers import MoleculeContainer, QueryContainer

from synplan.chem.utils import safe_canonicalization
from synplan.enumeration.synthon.config import SynthonConfig, load_data


class SynthonDataError(ValueError):
    """A shipped pattern failed to compile. Never log-and-skip: 93% of the corpus is exclusions,
    and a dropped exclusion over-classifies, which no smoke test can see."""


def _compile(patterns: list[str], where: str) -> list[QueryContainer]:
    out = []
    for pattern in patterns:
        try:
            out.append(smarts(pattern))
        except Exception as exc:
            raise SynthonDataError(f"{where}: {pattern!r}") from exc
    return out


class BBClassifier:
    """`any(at_least_one) and all(also) and not any(not)` over an ORDERED class list.

    Order is load-bearing: the synthoniser breaks on the first poly-functional class and composes
    mono classes in file order. Never sort, never key by name.
    """

    def __init__(self, config: SynthonConfig | None = None) -> None:
        self.config = config or SynthonConfig()
        records = load_data(self.config.classes_path)
        if len(records) != 147:
            raise SynthonDataError(
                f"{len(records)} classes in {self.config.classes_path}, expected 147"
            )
        self.classes = [
            (
                record["name"],
                _compile(record["at_least_one"], f"{record['name']}/at_least_one"),
                _compile(record["also"], f"{record['name']}/also"),
                _compile(record["not"], f"{record['name']}/not"),
            )
            for record in records
        ]

    def classify(self, molecule: MoleculeContainer) -> list[str]:
        """Classes of an ALREADY canonicalised molecule. chython does not aromatise on parse, so a
        kekule input silently matches nothing."""
        out = []
        for name, at_least_one, also, excluded in self.classes:
            if not any(q.is_substructure(molecule) for q in at_least_one):
                continue
            if not all(q.is_substructure(molecule) for q in also):
                continue
            if any(q.is_substructure(molecule) for q in excluded):
                continue
            out.append(name)
        return out

    def classify_smiles(self, smi: str) -> list[str] | None:
        """Every `.`-separated component classified on its own, unioned back into class order.

        Whole-string matching pairs a group in one component against a group in another and
        invents a class, and one counter-ion's exclusion patterns wipe out the parent's classes.
        None when nothing in the row parses.
        """
        found: set[str] = set()
        parsed = False
        for part in smi.split("."):
            try:
                molecule = safe_canonicalization(smiles(part))
            except Exception:
                continue
            parsed = True
            found.update(self.classify(molecule))
        if not parsed:
            return None
        return [name for name, *_ in self.classes if name in found]


__all__ = ["BBClassifier", "SynthonDataError"]
