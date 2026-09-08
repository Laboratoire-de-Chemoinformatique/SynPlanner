"""Real USPTO outcomes hidden by mapping truncation or query symmetry."""

import json
from pathlib import Path

import pytest

from synplan.chem.precursor import Precursor
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import RolloutEvaluationConfig, TreeConfig
from synplan.mcts.evaluation import RandomEvaluationStrategy
from synplan.mcts.tree import Tree
from synplan.utils.loading import load_evaluation_function, load_reaction_rules

CASES = json.loads(
    (Path(__file__).parents[1] / "data/regression/reaction_outcomes.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["step"]))
@pytest.mark.parametrize("engine", ["tree", "rollout"])
def test_configured_outcome_budget_and_selective_symmetry(case, engine, tmp_path):
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
        target = mol_from_smiles(case["product"], clean2d=False)
        if engine == "rollout":
            evaluator = load_evaluation_function(
                RolloutEvaluationConfig(
                    policy_network=Policy(),
                    reaction_rules=rules,
                    building_blocks=set(),
                    max_reaction_outcomes=budget,
                )
            )
            return {
                tuple(sorted(str(product) for product in products))
                for products in evaluator.rollout._apply_rule(
                    Precursor(target), rules[0]
                )
            }
        tree = Tree(
            target,
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

    small = outcomes(case["below_budget"])
    complete = outcomes(case["budget"])
    # Bounded traversal can discover the recorded outcome earlier than the old
    # native matcher. The configured limit and retained chemistry are the API.
    assert len(small) <= case["below_budget"]
    assert small <= complete
    assert expected in complete
