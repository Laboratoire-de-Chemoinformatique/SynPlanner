"""Detached-Route integration tests for ReTReK scoring."""

from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.reaction.reactor import CanonicalRetroReactor, Reaction
from synplan.chem.reaction.routes.quality.retrek import (
    ASRouteScorer,
    RDRouteScorer,
    RetrekRouteScorer,
    RetrekRouteScoringConfig,
    STRouteScorer,
)
from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step, StepOrigin
from synplan.chem.utils import molecule_key


def _one_step_route(*, search_score=None):
    available = smiles("CC")
    unavailable = smiles("CO")
    target = smiles("CCO")
    reaction = Reaction([available, unavailable], [target])
    return (
        Route(
            steps=(Step(reaction, target),),
            provenance=RouteProvenance(search_score=search_score),
        ),
        available,
        unavailable,
    )


def test_route_context_uses_step_product_not_the_first_reaction_product():
    reactants = [smiles("CCC"), smiles("CCC")]
    ring_byproduct = smiles("C1CCCCC1")
    focus = smiles("CCCCCC")
    reaction = Reaction(reactants, [ring_byproduct, focus])
    route = Route(steps=(Step(reaction, focus),))

    assert RDRouteScorer().score(route) == 0.0


def test_asscore_reads_leaf_availability_from_the_route():
    route, _, unavailable = _one_step_route()
    object.__setattr__(route, "unresolved", frozenset({molecule_key(unavailable)}))

    assert ASRouteScorer().score(route) == 0.5


def test_asscore_counts_route_leaves_but_not_synthesized_intermediates():
    first_leaf = smiles("CC")
    second_leaf = smiles("CO")
    final_leaf = smiles("N")
    intermediate = smiles("CCO")
    target = smiles("CCON")

    make_intermediate = Reaction([first_leaf, second_leaf], [intermediate])
    make_target = Reaction([intermediate, final_leaf], [target])
    route = Route(
        steps=(
            Step(make_intermediate, intermediate),
            Step(make_target, target),
        )
    )

    assert ASRouteScorer().step_scores(route) == (1.0, 0.5)


def test_route_score_is_independent_of_search_provenance():
    low, _, _ = _one_step_route(search_score=0.1)
    high, _, _ = _one_step_route(search_score=0.9)
    scorer = RetrekRouteScorer()

    assert scorer.score(low) == scorer.score(high)


def test_route_scorer_ranks_route_objects_by_its_own_score():
    solved, _, _ = _one_step_route()
    unsolved, _, unavailable = _one_step_route()
    object.__setattr__(unsolved, "unresolved", frozenset({molecule_key(unavailable)}))

    assert ASRouteScorer().rank([unsolved, solved]) == [solved, unsolved]


def test_stscore_requires_reaction_rules():
    with pytest.raises(ValueError, match="reaction_rules"):
        STRouteScorer()


def test_stscore_indexes_reaction_rules_with_the_step_rule_id():
    precursor = smiles("CC")
    target = smiles("CCO")
    reaction = Reaction([precursor], [target])
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]")
    route = Route(
        steps=(Step(reaction, target, StepOrigin(rule_id=1)),),
    )

    assert STRouteScorer(reaction_rules=(object(), rule)).score(route) == 1.0


def test_example_config_loads_with_route_defaults():
    path = Path(__file__).resolve().parents[3] / "configs/retrek_route_quality.yaml"

    config = RetrekRouteScoringConfig.from_yaml(path)

    assert config.enabled_scores == ("cd", "as", "rd")
    assert RetrekRouteScorer(config)


def test_retrek_route_scorer_is_available_from_the_lazy_route_facade():
    from synplan.chem.reaction.routes import RetrekRouteScorer as FacadeScorer

    assert FacadeScorer is RetrekRouteScorer


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"enabled_scores": (), "weights": {}}, "one"),
        ({"enabled_scores": ("cd", "cd"), "weights": {"cd": 1.0}}, "unique"),
        ({"enabled_scores": ("cd",), "weights": {}}, "missing"),
        ({"enabled_scores": ("cd",), "weights": {"cd": -1.0}}, "non-negative"),
        ({"enabled_scores": ("cd",), "weights": {"cd": 0.0}}, "positive"),
    ],
)
def test_config_rejects_invalid_score_selections(values, message):
    with pytest.raises(ValueError, match=message):
        RetrekRouteScoringConfig(**values)
