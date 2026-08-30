"""Detached-Route integration tests for ReTReK scoring."""

from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.quality.retrek import (
    ASRouteScorer,
    RDRouteScorer,
    RetrekRouteScorer,
    RetrekRouteScoringConfig,
    STRouteScorer,
)
from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step
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


def test_stscore_requires_an_explicit_step_resolver():
    with pytest.raises(ValueError, match="rule_resolver"):
        STRouteScorer()


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
