"""Tests for tree statistics collection."""

from collections.abc import Callable

from chython.containers import MoleculeContainer

from synplan.mcts.tree import Tree
from synplan.utils.config import RolloutEvaluationConfig, TreeConfig
from synplan.utils.loading import load_evaluation_function

# -- Helpers (same pattern as test_algorithm.py) --


def make_mol(n: int) -> MoleculeContainer:
    m = MoleculeContainer()
    prev = None
    for _ in range(n):
        a = m.add_atom("C")
        if prev is not None:
            m.add_bond(prev, a, 1)
        prev = a
    return m


class FakeReaction:
    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products = products


class FakeReactor:
    def __init__(self, products_fn: Callable[[], list[MoleculeContainer]]):
        self.products_fn = products_fn

    def __call__(self, *reactants: MoleculeContainer):
        return [FakeReaction(list(reactants), self.products_fn())]


class FakePolicy:
    def __init__(self, rules, expand_deeper=False):
        self.rules = rules
        self.expand_deeper = expand_deeper

    def predict_reaction_rules(self, precursor, reaction_rules):
        if not self.expand_deeper and len(precursor.prev_precursors) > 1:
            return
        for prob, reactor, rid in self.rules:
            yield prob, reactor, rid


def build_tree(algorithm="breadth_first", rules=None, **kwargs):
    if rules is None:
        rules = [(0.5, FakeReactor(lambda: [make_mol(5)]), 0)]
    expand_deeper = kwargs.pop("expand_deeper", False)
    max_iterations = kwargs.pop("max_iterations", 20)
    cfg = TreeConfig(
        algorithm=algorithm,
        max_iterations=max_iterations,
        max_tree_size=200,
        max_time=10,
        max_depth=4,
        search_strategy="expansion_first",
        min_mol_size=6,
        silent=True,
        enable_pruning=False,
        **kwargs,
    )
    target = make_mol(7)
    fake_policy = FakePolicy(rules, expand_deeper=expand_deeper)
    reactors = [r for _, r, _ in rules]
    eval_config = RolloutEvaluationConfig(
        policy_network=fake_policy, reaction_rules=reactors, building_blocks=set()
    )
    evaluator = load_evaluation_function(eval_config)
    return Tree(
        target=target,
        config=cfg,
        reaction_rules=reactors,
        building_blocks=set(),
        expansion_function=fake_policy,
        evaluation_function=evaluator,
    )


# -- Task 1: Stats dict exists --


def test_stats_dict_exists_on_new_tree():
    tree = build_tree()
    assert hasattr(tree, "stats")
    assert isinstance(tree.stats, dict)
    assert tree.stats["expansion_calls"] == 0
    assert tree.stats["expansion_successes"] == 0
    assert tree.stats["total_rules_tried"] == 0
    assert tree.stats["total_rules_succeeded"] == 0
    assert tree.stats["dead_end_nodes"] == 0
    assert tree.stats["first_solution_iteration"] is None
    assert tree.stats["first_solution_time"] is None
    assert tree.stats["routes_found_at"] == []


# -- Task 2: Expansion stats --


def test_expansion_stats_after_search():
    """After a search with solvable rules, stats should reflect expansions."""
    rules = [
        (0.8, FakeReactor(lambda: [make_mol(5)]), 0),  # building block (solved)
        (0.2, FakeReactor(lambda: [make_mol(10)]), 1),  # unsolved
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    assert tree.stats["expansion_calls"] >= 1
    assert tree.stats["total_rules_tried"] >= 2  # at least both rules tried on root
    assert tree.stats["total_rules_succeeded"] >= 1


def test_dead_end_counted():
    """A rule returning empty products on root raises StopIteration.
    Root dead-end is not counted as dead_end_nodes (it raises instead)."""
    rules = [
        (0.5, FakeReactor(lambda: []), 0),
    ]
    tree = build_tree("breadth_first", rules)
    try:
        for _ in tree:
            pass
    except StopIteration:
        pass

    assert tree.stats["expansion_calls"] >= 1
    # Root node dead-end raises StopIteration, doesn't increment dead_end_nodes
    assert tree.stats["dead_end_nodes"] == 0


# -- Task 3: Route discovery timing --


def test_route_discovery_timing():
    """When a route is found, first_solution_iteration and routes_found_at are set."""
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),  # building block -> solved
    ]
    tree = build_tree("breadth_first", rules)
    for solved, node_ids in tree:
        pass

    assert tree.stats["first_solution_iteration"] is not None
    assert tree.stats["first_solution_iteration"] >= 1
    assert tree.stats["first_solution_time"] is not None
    assert tree.stats["first_solution_time"] >= 0.0
    assert len(tree.stats["routes_found_at"]) >= 1
    entry = tree.stats["routes_found_at"][0]
    assert len(entry) == 2
    assert entry[0] >= 1  # iteration
    assert entry[1] >= 0.0  # time


def test_no_route_timing_when_unsolved():
    """When no route is found, first_solution stays None."""
    rules = [
        (0.5, FakeReactor(lambda: [make_mol(10)]), 0),  # unsolved
    ]
    tree = build_tree("breadth_first", rules, max_iterations=3)
    try:
        for _ in tree:
            pass
    except StopIteration:
        pass

    assert tree.stats["first_solution_iteration"] is None
    assert tree.stats["first_solution_time"] is None
    assert tree.stats["routes_found_at"] == []


# -- Task 4: Winning rule ranks --


def test_winning_rule_ranks():
    """winning_rule_ranks returns info about rules on winning routes."""
    rules = [
        (0.8, FakeReactor(lambda: [make_mol(5)]), 0),  # solved
        (0.2, FakeReactor(lambda: [make_mol(10)]), 1),  # unsolved
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    assert len(tree.winning_nodes) > 0
    ranks = tree.winning_rule_ranks()
    assert isinstance(ranks, list)
    for route_info in ranks:
        assert "winning_node_id" in route_info
        assert "steps" in route_info
        assert isinstance(route_info["steps"], list)
        for step in route_info["steps"]:
            assert "node_id" in step
            assert "rule_id" in step
            assert "prob" in step
            assert "rank" in step
            assert step["rank"] >= 1


# -- Task 5: Applicability rate and branching --


def test_rule_applicability_rate():
    rules = [
        (0.8, FakeReactor(lambda: [make_mol(5)]), 0),
        (0.2, FakeReactor(lambda: [make_mol(10)]), 1),
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    rate = tree.rule_applicability_rate()
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


def test_rule_applicability_rate_no_expansions():
    """Before any search, rate should be 0.0."""
    tree = build_tree()
    assert tree.rule_applicability_rate() == 0.0


def test_branching_profile():
    rules = [
        (0.8, FakeReactor(lambda: [make_mol(5)]), 0),
        (0.2, FakeReactor(lambda: [make_mol(10)]), 1),
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    profile = tree.branching_profile()
    assert isinstance(profile, dict)
    assert 0 in profile
    assert "mean_children" in profile[0]
    assert "nodes" in profile[0]
    assert profile[0]["nodes"] >= 1


# -- Task 6: Route details --


def test_route_details():
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    assert len(tree.winning_nodes) > 0
    details = tree.route_details(tree.winning_nodes[0])
    assert isinstance(details, dict)
    assert "node_id" in details
    assert "route_score" in details
    assert "route_length" in details
    assert "steps" in details
    assert isinstance(details["steps"], list)


# -- Task 7: to_stats_dict --


def test_to_stats_dict():
    rules = [
        (0.8, FakeReactor(lambda: [make_mol(5)]), 0),
        (0.2, FakeReactor(lambda: [make_mol(10)]), 1),
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass

    d = tree.to_stats_dict()
    assert isinstance(d, dict)

    expected_keys = {
        "num_routes",
        "num_nodes",
        "num_iter",
        "tree_depth",
        "search_time",
        "solved",
        "expansion_calls",
        "expansion_successes",
        "total_rules_tried",
        "total_rules_succeeded",
        "rule_applicability_rate",
        "dead_end_nodes",
        "first_solution_iteration",
        "first_solution_time",
        "max_branching_factor",
        "mean_branching_factor",
    }
    assert expected_keys.issubset(d.keys()), f"Missing keys: {expected_keys - d.keys()}"
