"""Real GPS ester outcomes must survive complete-set deduplication."""

import json
from pathlib import Path

import pytest

from synplan.chem.reaction.reactor import ReactionApplication
from synplan.chem.target_bonds import TargetAtomProvenance
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RandomEvaluationStrategy
from synplan.mcts.tree import Tree, _ExpansionContext, _RuleCandidate
from synplan.utils.loading import load_reaction_rules

CASES = json.loads(
    (Path(__file__).parents[1] / "data/regression/precursor_multisets.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["target_id"])
def test_ester_only_outcome_survives_ester_with_hydroxide(case, tmp_path):
    path = tmp_path / "rules.tsv"
    path.write_text(
        "rule_smarts\tpopularity\n"
        + "".join(f"{rule['smarts']}\t1\n" for rule in case["rules"])
    )
    rules = load_reaction_rules(path)

    class Policy:
        def predict_reaction_rules(self, precursor, reaction_rules):
            for index, rule in enumerate(rules):
                yield case["rules"][index]["probability"], rule, index

    tree = Tree(
        target=mol_from_smiles(case["product"], clean2d=False),
        config=TreeConfig(min_mol_size=0, silent=True),
        reaction_rules=rules,
        building_blocks=set(),
        expansion_function=Policy(),
        evaluation_function=RandomEvaluationStrategy(),
    )
    tree._expand_node(1)
    outcomes = {
        tuple(sorted(str(p.molecule) for p in tree.nodes[child].new_precursors))
        for child in tree.children[1]
    }
    expected = tuple(sorted(case["expected_precursors"]))
    assert expected in outcomes
    assert tuple(sorted([*expected, "[OH-]"])) in outcomes
    assert len(outcomes) == len(case["rules"])


def test_multiplicity_and_rejected_cycles_do_not_poison_dedup():
    class Policy:
        def predict_reaction_rules(self, *args):
            return iter(())

    target = mol_from_smiles("CCCCCC", clean2d=False)
    tree = Tree(
        target,
        TreeConfig(min_mol_size=0),
        [],
        set(),
        Policy(),
        RandomEvaluationStrategy(),
    )
    context = _ExpansionContext(1, tree.nodes[1], [tree.nodes[1].curr_precursor], set())
    candidate = _RuleCandidate(0.5, None, 0, "policy", 1)
    ethanol = mol_from_smiles("CCO", clean2d=False)

    def application(*products):
        return ReactionApplication(products, (TargetAtomProvenance(),) * len(products))

    assert not tree._add_child_if_new(context, application(target, ethanol), candidate)
    assert not context.seen_products
    assert tree._add_child_if_new(context, application(ethanol), candidate)
    assert tree._add_child_if_new(
        context, application(ethanol.copy(), ethanol.copy()), candidate
    )
    assert not tree._add_child_if_new(context, application(ethanol.copy()), candidate)
