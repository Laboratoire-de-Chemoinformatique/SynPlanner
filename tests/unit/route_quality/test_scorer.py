"""Tests for the competing sites scorer and route re-ranking module."""

from unittest.mock import MagicMock

import pytest
from chython import smiles

from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.quality.protection.config import ProtectionConfig
from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.chem.reaction.routes.quality.protection.scanner import (
    CompetingInteraction,
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.chem.reaction.routes.quality.protection.scorer import CompetingSitesScore
from synplan.chem.reaction.routes.quality.scorer import (
    ProtectionRouteScorer,
    RouteScorer,
)
from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step


@pytest.fixture
def config():
    return ProtectionConfig()


@pytest.fixture
def matrix(config):
    return IncompatibilityMatrix(config.incompatibility_path)


@pytest.fixture
def fg_detector(config):
    return FunctionalGroupDetector(config.competing_groups_path)


@pytest.fixture
def halogen_detector(config):
    return HalogenDetector(config.halogen_groups_path)


@pytest.fixture
def scanner(fg_detector, matrix):
    return RouteScanner(fg_detector, matrix)


@pytest.fixture
def scanner_with_halogens(fg_detector, matrix, halogen_detector):
    return RouteScanner(fg_detector, matrix, halogen_detector=halogen_detector)


@pytest.fixture
def scorer(scanner):
    return CompetingSitesScore(scanner)


@pytest.fixture
def scorer_with_halogens(scanner_with_halogens):
    return CompetingSitesScore(scanner_with_halogens)


# --- score_route tests ---


def test_score_perfect_route_no_competing_sites(scorer):
    """Route with no competing FGs should have S(T) = 1.0."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    route = {0: rxn}
    score, interactions = scorer.score_route(route)
    assert score == 1.0
    assert len(interactions) == 0


def test_score_route_with_incompatible_fgs(scorer):
    """Route with incompatible FGs should have S(T) <= 1.0."""
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    route = {0: rxn}
    score, interactions = scorer.score_route(route)
    assert 0.0 <= score <= 1.0
    # Note: interactions can all be "compatible" severity (penalty=0),
    # so score can be 1.0 even with detected FGs.
    non_compatible = [i for i in interactions if i.severity != "compatible"]
    if len(non_compatible) > 0:
        assert score < 1.0


def test_score_empty_route(scorer):
    """Empty route should give S(T) = 1.0 (no interactions)."""
    score, interactions = scorer.score_route({})
    assert score == 1.0
    assert len(interactions) == 0


def test_score_is_bounded_zero_to_one(scorer):
    """Score should always be in [0, 1]."""
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    route = {0: rxn}
    score, _ = scorer.score_route(route)
    assert 0.0 <= score <= 1.0


def test_score_maximally_conflicting_route():
    """Route with many incompatible interactions should have S(T) close to 0."""
    mock_scanner = MagicMock(spec=RouteScanner)
    mock_scanner.scan_route.return_value = (
        [
            CompetingInteraction(
                step_id=0,
                fg_name="hydroxyl",
                fg_atoms=(1,),
                reacting_fg="aldehyde",
                severity="incompatible",
            ),
            CompetingInteraction(
                step_id=0,
                fg_name="primary_amine",
                fg_atoms=(2,),
                reacting_fg="aldehyde",
                severity="incompatible",
            ),
            CompetingInteraction(
                step_id=0,
                fg_name="thiol",
                fg_atoms=(3,),
                reacting_fg="aldehyde",
                severity="incompatible",
            ),
        ],
        0,  # halogen_count
    )
    mock_scanner.last_processed_steps = {0}
    scorer = CompetingSitesScore(mock_scanner)
    route = {0: MagicMock()}
    score, interactions = scorer.score_route(route)
    # worst-per-step: step 0 worst = 1.0 -> penalty = 1.0/1 -> score = 0.0
    assert score == 0.0
    assert len(interactions) == 3


def test_score_with_competing_severity():
    """Competing interactions contribute worst-per-step penalty."""
    mock_scanner = MagicMock(spec=RouteScanner)
    mock_scanner.scan_route.return_value = (
        [
            CompetingInteraction(
                step_id=0,
                fg_name="hydroxyl",
                fg_atoms=(1,),
                reacting_fg="aldehyde",
                severity="competing",
            ),
            CompetingInteraction(
                step_id=0,
                fg_name="phenol",
                fg_atoms=(2,),
                reacting_fg="aldehyde",
                severity="competing",
            ),
        ],
        0,
    )
    mock_scanner.last_processed_steps = {0}
    scorer = CompetingSitesScore(mock_scanner)
    route = {0: MagicMock()}
    score, _ = scorer.score_route(route)
    # worst-per-step: step 0 worst = 0.5 -> penalty = 0.5/1 -> score = 0.5
    assert score == pytest.approx(0.5)


def test_score_mixed_severities():
    """Mixed incompatible + competing gives expected score."""
    mock_scanner = MagicMock(spec=RouteScanner)
    mock_scanner.scan_route.return_value = (
        [
            CompetingInteraction(
                step_id=0,
                fg_name="hydroxyl",
                fg_atoms=(1,),
                reacting_fg="aldehyde",
                severity="incompatible",
            ),
            CompetingInteraction(
                step_id=1,
                fg_name="phenol",
                fg_atoms=(2,),
                reacting_fg="ketone",
                severity="competing",
            ),
        ],
        0,
    )
    mock_scanner.last_processed_steps = {0, 1}
    scorer = CompetingSitesScore(mock_scanner)
    route = {0: MagicMock(), 1: MagicMock()}
    score, _ = scorer.score_route(route)
    # worst-per-step: step 0 = 1.0, step 1 = 0.5 -> penalty = 1.5/2 = 0.75
    assert score == pytest.approx(0.25)


def test_score_with_halogen_count():
    """Halogen count H should contribute to the penalty."""
    mock_scanner = MagicMock(spec=RouteScanner)
    mock_scanner.scan_route.return_value = (
        [
            CompetingInteraction(
                step_id=0,
                fg_name="hydroxyl",
                fg_atoms=(1,),
                reacting_fg="aldehyde",
                severity="incompatible",
            ),
        ],
        2,  # halogen_count
    )
    mock_scanner.last_processed_steps = {0}
    scorer = CompetingSitesScore(mock_scanner)
    route = {0: MagicMock()}
    score, _ = scorer.score_route(route)
    # worst-per-step: step 0 = 1.0, + halogen 2 -> penalty = 3.0/1 -> 0.0
    assert score == 0.0


def test_score_with_halogen_count_partial():
    """Halogen count adds partial penalty."""
    mock_scanner = MagicMock(spec=RouteScanner)
    mock_scanner.scan_route.return_value = (
        [],  # no FG interactions
        1,  # 1 halogen
    )
    mock_scanner.last_processed_steps = {0, 1}
    scorer = CompetingSitesScore(mock_scanner)
    route = {0: MagicMock(), 1: MagicMock()}
    score, _ = scorer.score_route(route)
    # penalty = (0 + 0 + 1) / 2 = 0.5 -> score = 0.5
    assert score == pytest.approx(0.5)


# --- ProtectionConfig tests ---


def test_config_defaults():
    """ProtectionConfig should have sensible defaults."""
    cfg = ProtectionConfig()
    assert isinstance(cfg.competing_groups_path, str)
    assert isinstance(cfg.incompatibility_path, str)
    assert isinstance(cfg.halogen_groups_path, str)


def test_config_from_dict():
    """ProtectionConfig.from_dict should create a valid config."""
    cfg = ProtectionConfig.from_dict(
        {
            "competing_groups_path": "/tmp/cg.yaml",
            "incompatibility_path": "/tmp/im.yaml",
            "halogen_groups_path": "/tmp/hal.yaml",
        }
    )
    assert cfg.competing_groups_path == "/tmp/cg.yaml"
    assert cfg.halogen_groups_path == "/tmp/hal.yaml"


def test_config_to_yaml_roundtrip(tmp_path):
    """Config should survive a to_yaml -> from_yaml roundtrip."""
    cfg = ProtectionConfig(halogen_groups_path="/tmp/hal.yaml")
    yaml_path = str(tmp_path / "rt.yaml")
    cfg.to_yaml(yaml_path)
    assert ProtectionConfig.from_yaml(yaml_path) == cfg


# --- RouteScorer: score and rank take Route objects ---

# Ethyl benzoate from benzoic acid and ethanol, atom maps shared as a reactor's are.
_TARGET = "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
_ACID = "[OH:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
_ETHANOL = "[CH3:1][CH2:2][OH:12]"


def _esterification(search_score):
    reaction = Reaction([smiles(_ACID), smiles(_ETHANOL)], [smiles(_TARGET)])
    return Route(
        steps=(Step(reaction, reaction.products[0]),),
        provenance=RouteProvenance(search_score),
    )


class _BySearchScore(RouteScorer):
    """A scorer whose verdict IS the search score, so rank's ordering is under test."""

    def score(self, route):
        return route.provenance.search_score


class _ByStepCount(RouteScorer):
    """A scorer that never looks at the search, which the base class must allow."""

    def score(self, route):
        return 1.0 / len(route)


def test_rank_orders_by_the_scorer_s_own_number():
    routes = [_esterification(0.1), _esterification(0.9), _esterification(0.5)]
    ranked = _BySearchScore().rank(routes)
    assert [r.provenance.search_score for r in ranked] == [0.9, 0.5, 0.1]


def test_a_scorer_that_ignores_the_search_needs_no_search_score():
    """The base class must not assume every verdict is a multiple of the search's."""
    good, bad = _esterification(None), _esterification(None)
    object.__setattr__(bad, "steps", bad.steps * 2)  # a worse route, same provenance
    assert _ByStepCount().rank([bad, good])[0] is good


def test_rank_of_nothing_is_nothing():
    assert _ByStepCount().rank([]) == []


def test_protection_scorer_scores_a_route():
    route = _esterification(0.5)
    scorer = ProtectionRouteScorer.from_config()
    assert 0.0 <= scorer.competing_sites_score(route) <= 1.0
    assert scorer.score(route) == pytest.approx(
        0.5 * scorer.competing_sites_score(route)
    )


def test_the_paper_s_score_refuses_a_route_with_no_search_behind_it():
    """search_score * S(T) is undefined without a search score; S(T) alone is not."""
    scorer = ProtectionRouteScorer.from_config()
    route = _esterification(None)
    with pytest.raises(ValueError, match="no search score"):
        scorer.score(route)
    assert 0.0 <= scorer.competing_sites_score(route) <= 1.0


def test_rerank_routes_basic(scorer):
    """rank_routes should sort by combined score descending."""
    rxn_a = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    rxn_b = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    routes = {0: {0: rxn_a}, 1: {0: rxn_b}}
    ranked = scorer.rerank_routes(routes)
    assert isinstance(ranked, list)
    assert len(ranked) == 2
    for entry in ranked:
        assert len(entry) == 4
    assert ranked[0][1] >= ranked[1][1]


def test_rerank_routes_with_existing_scores(scorer):
    """rank_routes should combine original scores with protection scores."""
    rxn_a = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    rxn_b = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    routes = {0: {0: rxn_a}, 1: {0: rxn_b}}
    existing = {0: 0.5, 1: 1.0}
    ranked = scorer.rerank_routes(routes, existing_scores=existing, weight=0.5)
    assert len(ranked) == 2
    route_ids = [r[0] for r in ranked]
    assert route_ids[0] == 1


def test_rerank_routes_no_existing_scores(scorer):
    """Without existing scores, ranking uses only protection scores."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    routes = {0: {0: rxn}}
    ranked = scorer.rerank_routes(routes)
    assert len(ranked) == 1
    _route_id, combined, protection, original = ranked[0]
    assert original == 0.0
    assert combined == pytest.approx(0.5 * protection)


def test_rerank_routes_weight_zero(scorer):
    """With weight=0, only original scores matter."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    routes = {0: {0: rxn}, 1: {0: rxn}}
    existing = {0: 0.3, 1: 0.9}
    ranked = scorer.rerank_routes(routes, existing_scores=existing, weight=0.0)
    route_ids = [r[0] for r in ranked]
    assert route_ids[0] == 1


def test_rerank_routes_weight_one(scorer):
    """With weight=1, only protection scores matter."""
    rxn_clean = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    rxn_dirty = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    routes = {0: {0: rxn_clean}, 1: {0: rxn_dirty}}
    existing = {0: 0.1, 1: 1.0}
    ranked = scorer.rerank_routes(routes, existing_scores=existing, weight=1.0)
    assert ranked[0][0] == 0


# --- ProtectionConfig tests ---
