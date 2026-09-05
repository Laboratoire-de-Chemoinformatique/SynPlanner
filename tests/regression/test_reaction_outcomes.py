"""Real USPTO outcomes hidden by mapping truncation or query symmetry."""

import json
from pathlib import Path

import pytest

from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RandomEvaluationStrategy
from synplan.mcts.tree import Tree
from synplan.utils.loading import load_reaction_rules

CASES = json.loads(
    (Path(__file__).parents[1] / "data/regression/reaction_outcomes.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["step"]))
def test_configured_outcome_budget_and_selective_symmetry(case, tmp_path):
    path = tmp_path / "rules.tsv"
    path.write_text(f"rule_smarts\tpopularity\n{case['smarts']}\t1\n")
    rules = load_reaction_rules(path)

    class Policy:
        def predict_reaction_rules(self, *args):
            yield 0.5, rules[0], 0

    expected = tuple(
        sorted(str(mol_from_smiles(s, clean2d=False)) for s in case["gold"])
    )

    def outcomes(budget):
        tree = Tree(
            mol_from_smiles(case["product"], clean2d=False),
            TreeConfig(max_reaction_outcomes=budget, min_mol_size=0, silent=True),
            rules,
            set(),
            Policy(),
            RandomEvaluationStrategy(),
        )
        tree._expand_node(1)
        return {
            tuple(sorted(str(p.molecule) for p in tree.nodes[child].new_precursors))
            for child in tree.children[1]
        }

    if case["step"] == 389:
        assert expected not in outcomes(5)
        assert expected in outcomes(20)
    else:
        assert expected in outcomes(5)
