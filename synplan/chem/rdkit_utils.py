from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.*")


class RDKitScore:
    """Node scoring function."""

    def __init__(self, score_function="heavyAtomCount") -> None:
        self.score_function = score_function
        # Normalization constants to bound outputs to [0, 1]
        self._H_MAX = 100.0
        self._W_MAX = 1000.0

    def __call__(self, node):

        if self.score_function == "sascore":
            meanPrecursorSAS = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    meanPrecursorSAS += sascorer.calculateScore(m)
                except:
                    meanPrecursorSAS += 10.0

            if (
                len(node.precursors_to_expand) == 0
            ):  # TODO ZeroDivisionError: division by zero
                return 0

            meanPrecursorSAS = meanPrecursorSAS / len(node.precursors_to_expand)
            node_value = 1.0 - meanPrecursorSAS / 10.0
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "heavyAtomCount":
            totalHeavy = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalHeavy += CalcNumHeavyAtoms(m)
                except:
                    totalHeavy += 100.0

            # Map to [0, 1]: higher heavy atom count -> lower score
            node_value = 1.0 - (totalHeavy / self._H_MAX)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "weight":
            totalWeight = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalWeight += ExactMolWt(m)
                except:
                    totalWeight += 1000.0

            # Map to [0, 1]: higher weight -> lower score
            node_value = 1.0 - (totalWeight / self._W_MAX)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "weightXsascore":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            # Smoothly bound to (0, 1]
            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value

        elif self.score_function == "WxWxSAS":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) ** 2 * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            node_value = 1.0 / (1.0 + total)
            if node_value < 0:
                node_value = 0.0
            if node_value > 1:
                node_value = 1.0

            return node_value
