from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class NodeScore:
    """Node scoring function."""

    def __init__(self, score_function="heavyAtomCount") -> None:
        self.score_function = score_function

    def __call__(self, node):

        if self.score_function == "sascore":
            meanPrecursorSAS = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    meanPrecursorSAS += sascorer.calculateScore(m)
                except:
                    meanPrecursorSAS += 10.0

            meanPrecursorSAS = meanPrecursorSAS / len(node.precursors_to_expand)
            node_value = 1.0-meanPrecursorSAS / 10.0

            return node_value

        elif self.score_function == "heavyAtomCount":
            totalHeavy = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalHeavy += CalcNumHeavyAtoms(m)
                except:
                    totalHeavy += 100.0

            node_value = 1000 - totalHeavy
            if node_value < 0:
                node_value = 0

            return node_value

        elif self.score_function == "weight":
            totalWeight = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalWeight += ExactMolWt(m)
                except:
                    totalWeight += 1000.0

            node_value = 10000 - totalWeight
            if node_value < 0:
                node_value = 0

            return node_value

        elif self.score_function == "weightXsascore":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            if total == 0:
                return 1
            node_value = 1 / total
            if node_value < 0:
                node_value = 0

            return node_value

        elif self.score_function == "WxWxSAS":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) ** 2 * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            if total == 0:
                return 1
            node_value = 1 / total
            if node_value < 0:
                node_value = 0

            return node_value