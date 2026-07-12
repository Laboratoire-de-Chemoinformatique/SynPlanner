from collections.abc import Callable

from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")


class RDKitScore:
    """Node scoring function."""

    _SUPPORTED_SCORES = frozenset(
        {"sascore", "heavyAtomCount", "weight", "weightXsascore", "WxWxSAS"}
    )

    def __init__(self, score_function="heavyAtomCount") -> None:
        self.score_function = score_function
        # Normalization constants to bound outputs to [0, 1]
        self._H_MAX = 100.0
        self._W_MAX = 1000.0

    @staticmethod
    def _parse_precursors(node) -> list:
        """Convert each precursor to RDKit once for the selected metric."""

        molecules = []
        for precursor in node.precursors_to_expand:
            try:
                molecule = Chem.MolFromSmiles(str(precursor.molecule))
            except Exception:
                molecule = None
            molecules.append(molecule)
        return molecules

    @staticmethod
    def _safe_sum(
        molecules: list,
        metric: Callable[[object], float],
        fallback: float,
    ) -> float:
        """Sum a metric while retaining the historical per-molecule fallback."""

        total = 0.0
        for molecule in molecules:
            if molecule is None:
                total += fallback
                continue
            try:
                total += metric(molecule)
            except Exception:
                total += fallback
        return total

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def __call__(self, node):
        if self.score_function not in self._SUPPORTED_SCORES:
            return None

        molecules = self._parse_precursors(node)

        if self.score_function == "sascore":
            if not molecules:
                return 0
            mean_precursor_sas = self._safe_sum(
                molecules, sascorer.calculateScore, 10.0
            ) / len(molecules)
            return self._clamp(1.0 - mean_precursor_sas / 10.0)

        if self.score_function == "heavyAtomCount":
            total_heavy = self._safe_sum(molecules, CalcNumHeavyAtoms, 100.0)
            return self._clamp(1.0 - total_heavy / self._H_MAX)

        if self.score_function == "weight":
            total_weight = self._safe_sum(molecules, ExactMolWt, 1000.0)
            return self._clamp(1.0 - total_weight / self._W_MAX)

        if self.score_function == "weightXsascore":
            total = self._safe_sum(
                molecules,
                lambda molecule: ExactMolWt(molecule)
                * sascorer.calculateScore(molecule),
                10000.0,
            )
            return self._clamp(1.0 / (1.0 + total))

        total = self._safe_sum(
            molecules,
            lambda molecule: ExactMolWt(molecule) ** 2
            * sascorer.calculateScore(molecule),
            10000.0,
        )
        return self._clamp(1.0 / (1.0 + total))
