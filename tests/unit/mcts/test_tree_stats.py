"""Tests for tree statistics collection."""

from collections.abc import Callable

import pytest
from chython.containers import MoleculeContainer

from synplan.mcts.tree import Tree, TreeStats
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


class _AlwaysMatches:
    def __lt__(self, other):
        return True


class FakePriorityReactor(FakeReactor):
    """Mimics chython.Reactor: keeps LHS query patterns on ``_patterns``."""

    def __init__(self, products_fn: Callable[[], list[MoleculeContainer]]):
        super().__init__(products_fn)
        self._patterns = (_AlwaysMatches(),)


class FakePolicy:
    def __init__(self, rules, expand_deeper=False):
        self.rules = rules
        self.expand_deeper = expand_deeper

    def predict_reaction_rules(self, precursor, reaction_rules):
        if not self.expand_deeper and len(precursor.prev_precursors) > 1:
            return
        yield from self.rules


def build_tree(algorithm="breadth_first", rules=None, **kwargs):
    if rules is None:
        rules = [(0.5, FakeReactor(lambda: [make_mol(5)]), 0)]
    expand_deeper = kwargs.pop("expand_deeper", False)
    max_iterations = kwargs.pop("max_iterations", 20)
    priority_rules = kwargs.pop("priority_rules", None)
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
        priority_rules=priority_rules,
    )


# -- Task 1: Stats dict exists --


def test_stats_dict_exists_on_new_tree():
    tree = build_tree()
    assert hasattr(tree, "stats")
    assert isinstance(tree.stats, TreeStats)
    assert tree.stats.expansion_calls == 0
    assert tree.stats.expansion_successes == 0
    assert tree.stats.total_rules_tried == 0
    assert tree.stats.total_rules_succeeded == 0
    assert tree.stats.policy_rules_tried == 0
    assert tree.stats.policy_rules_succeeded == 0
    assert tree.stats.priority_rules_tried == 0
    assert tree.stats.priority_rules_succeeded == 0
    assert tree.stats.dead_end_nodes == 0
    assert tree.stats.first_solution_iteration is None
    assert tree.stats.first_solution_time is None
    assert tree.stats.routes_found_at == []


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

    assert tree.stats.expansion_calls >= 1
    assert tree.stats.total_rules_tried >= 2  # at least both rules tried on root
    assert tree.stats.total_rules_succeeded >= 1


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

    assert tree.stats.expansion_calls >= 1
    # Root node dead-end raises StopIteration, doesn't increment dead_end_nodes
    assert tree.stats.dead_end_nodes == 0


# -- Task 3: Route discovery timing --


def test_route_discovery_timing():
    """When a route is found, first_solution_iteration and routes_found_at are set."""
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),  # building block -> solved
    ]
    tree = build_tree("breadth_first", rules)
    for _solved, _node_ids in tree:
        pass

    assert tree.stats.first_solution_iteration is not None
    assert tree.stats.first_solution_iteration >= 1
    assert tree.stats.first_solution_time is not None
    assert tree.stats.first_solution_time >= 0.0
    assert len(tree.stats.routes_found_at) >= 1
    entry = tree.stats.routes_found_at[0]
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

    assert tree.stats.first_solution_iteration is None
    assert tree.stats.first_solution_time is None
    assert tree.stats.routes_found_at == []


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
            assert "rule_source" in step
            assert "rule_key" in step
            assert "prob" in step
            assert "rank" in step
            assert step["rank"] >= 1
            assert step["rule_source"] == "policy"
            assert step["rule_key"] == f"policy:{step['rule_id']}"


def test_priority_rule_metadata_and_counters():
    priority_rule = FakePriorityReactor(lambda: [make_mol(5)])
    tree = build_tree(
        "breadth_first",
        rules=[],
        priority_rules={"priority": [priority_rule]},
        use_priority=True,
    )
    for _ in tree:
        pass

    assert tree.stats.priority_rules_tried >= 1
    assert tree.stats.priority_rules_succeeded >= 1
    assert tree.stats.policy_rules_tried == 0
    assert tree.stats.policy_rules_succeeded == 0
    assert tree.stats.total_rules_tried == tree.stats.priority_rules_tried
    assert tree.stats.total_rules_succeeded == tree.stats.priority_rules_succeeded

    ranks = tree.winning_rule_ranks()
    assert len(ranks) >= 1
    step = ranks[0]["steps"][0]
    assert step["rule_source"] == "priority"
    assert step["rule_key"] == "priority:0"

    details = tree.route_details(tree.winning_nodes[0])
    assert details["steps"][0]["rule_source"] == "priority"
    assert details["steps"][0]["rule_key"] == "priority:0"

    stats = tree.to_stats_dict()
    assert stats["n_routes_with_priority"] == len(tree.winning_nodes)
    assert stats["fraction_routes_with_priority"] == 1.0


# -- Multi-source priority sets --


def test_multi_source_priority_dispatch():
    """Two named priority sets get separate per-source counters; aggregates sum."""
    ugi_rule = FakePriorityReactor(lambda: [make_mol(5)])
    boc_rule = FakePriorityReactor(lambda: [make_mol(7)])
    tree = build_tree(
        "breadth_first",
        rules=[],
        priority_rules={"ugi": [ugi_rule], "boc": [boc_rule]},
        use_priority=True,
    )
    for _ in tree:
        pass

    per_source = tree.stats.per_priority_source
    assert "ugi" in per_source
    assert "boc" in per_source
    assert per_source["ugi"].tried >= 1
    assert per_source["boc"].tried >= 1
    # Aggregates sum across all priority sets.
    assert tree.stats.priority_rules_tried == (
        per_source["ugi"].tried + per_source["boc"].tried
    )
    assert tree.stats.priority_rules_succeeded == (
        per_source["ugi"].succeeded + per_source["boc"].succeeded
    )

    # to_stats_dict() emits the per-source breakdown as a plain dict
    # of dicts, ready for JSON.
    stats = tree.to_stats_dict()
    assert stats["per_priority_source"]["ugi"]["tried"] == per_source["ugi"].tried


def test_use_priority_without_rules_raises():
    """Lighting up the flag with no rules is a silent footgun — fail loud.

    Authoring-time checks (reserved name, empty set, non-string key) live in
    ``parse_priority_rules`` and are exercised in
    ``tests/unit/chem/reaction_rules/test_priority.py``.
    """
    with pytest.raises(ValueError, match="use_priority"):
        build_tree(use_priority=True, priority_rules=None)


def test_priority_node_prob_scales_with_fragment_count():
    """An N-fragment priority disconnect enters UCB with prior N (>1.0).

    This pins the documented behaviour from Tree.__init__ docstring:
    priority rules enter with prob=1.0, then ``scaled_prob = prob × n``
    where ``n`` is the count of qualifying fragments.
    """

    # Three fragments, all above min_mol_size=6 → scaled_prob == 3.
    def three_frags():
        return [make_mol(8), make_mol(9), make_mol(10)]

    priority_rule = FakePriorityReactor(three_frags)
    tree = build_tree(
        "breadth_first",
        rules=[],
        priority_rules={"priority": [priority_rule]},
        use_priority=True,
    )
    for _ in tree:
        pass

    # The first child (id=2) was produced by the priority rule on the root.
    assert tree.nodes[2].prob == 3.0


def test_priority_first_then_policy_dedup_is_order_dependent():
    """When priority and policy yield the same product set on the same node,
    priority "claims" the dedup slot and the policy succeeded counter is not
    incremented for that product. Pin this so the behaviour change is visible.
    """
    same_products = lambda: [make_mol(5)]  # noqa: E731 — match for both sources
    policy_rule = FakeReactor(same_products)
    priority_rule = FakePriorityReactor(same_products)

    tree = build_tree(
        "breadth_first",
        rules=[(0.5, policy_rule, 0)],
        priority_rules={"priority": [priority_rule]},
        use_priority=True,
    )
    for _ in tree:
        pass

    # Both sources are tried on every expansion, so tried counts grow with
    # iterations. But only the priority source registers as succeeded
    # because it iterates first and fills tmp_products.
    assert tree.stats.priority_rules_tried >= 1
    assert tree.stats.priority_rules_succeeded >= 1
    assert tree.stats.policy_rules_tried >= 1
    assert tree.stats.policy_rules_succeeded == 0


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
        "n_routes_with_priority",
        "fraction_routes_with_priority",
        "num_nodes",
        "num_iter",
        "tree_depth",
        "search_time",
        "solved",
        "expansion_calls",
        "expansion_successes",
        "total_rules_tried",
        "total_rules_succeeded",
        "policy_rules_tried",
        "policy_rules_succeeded",
        "priority_rules_tried",
        "priority_rules_succeeded",
        "rule_applicability_rate",
        "dead_end_nodes",
        "first_solution_iteration",
        "first_solution_time",
        "max_branching_factor",
        "mean_branching_factor",
        "best_route_score",
        "mean_winning_rule_rank",
    }
    assert expected_keys.issubset(d.keys()), f"Missing keys: {expected_keys - d.keys()}"
    assert d["n_routes_with_priority"] == 0
    assert d["fraction_routes_with_priority"] == 0.0


# -- Task 8: Per-node attribute migration (1.5.0) --


def test_root_node_zero_initialised_state():
    """Root node has zero-initialised search state."""
    tree = build_tree()
    root = tree.nodes[1]
    assert root.visit == 0
    assert root.depth == 0
    assert root.prob == 0.0
    assert root.init_value == 0.0
    assert root.total_value == 0.0
    assert root.rule_id is None
    assert root.rule_source is None
    assert root.rule_key is None
    assert root.policy_rank is None


def test_child_depth_follows_parent():
    """A child added by _add_node has depth == parent.depth + 1."""
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),
    ]
    tree = build_tree("breadth_first", rules)
    for _ in tree:
        pass
    # All non-root nodes must satisfy the depth invariant
    for nid, node in tree.nodes.items():
        if nid == 1:
            continue
        parent_id = tree.parents[nid]
        assert node.depth == tree.nodes[parent_id].depth + 1


def test_backprop_mutates_node_visit_in_place():
    """UCT backprop increments node.visit on every node along the path."""
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),
    ]
    tree = build_tree("uct", rules, max_iterations=5)
    for _ in tree:
        pass
    # Root visit must reflect the iteration count of UCT _update_visits paths.
    assert tree.nodes[1].visit >= 1


# -- Task 9: Loud-error contract (1.5.0) --


@pytest.mark.parametrize(
    "removed_name,replacement_fragment",
    [
        ("nodes_visit", "tree.nodes[node_id].visit"),
        ("nodes_depth", "tree.nodes[node_id].depth"),
        ("nodes_prob", "tree.nodes[node_id].prob"),
        ("nodes_init_value", "tree.nodes[node_id].init_value"),
        ("nodes_total_value", "tree.nodes[node_id].total_value"),
        ("nodes_rules", "tree.nodes[node_id].rule_id"),
    ],
)
def test_removed_node_attr_raises_with_migration_hint(
    removed_name, replacement_fragment
):
    """Reading any of the removed nodes_* attrs raises AttributeError with hint."""
    tree = build_tree()
    with pytest.raises(AttributeError) as excinfo:
        getattr(tree, removed_name)
    msg = str(excinfo.value)
    assert removed_name in msg
    assert "1.5.0" in msg
    assert replacement_fragment in msg


def test_unknown_tree_attribute_falls_through():
    """Unknown attributes raise plain AttributeError (no migration text)."""
    tree = build_tree()
    with pytest.raises(AttributeError) as excinfo:
        _ = tree.does_not_exist
    msg = str(excinfo.value)
    assert "1.5.0" not in msg
    assert "tree.nodes[node_id]" not in msg


def test_tree_stats_subscript_on_known_field_raises_typeerror():
    """tree.stats["expansion_calls"] raises TypeError with attribute-style hint."""
    tree = build_tree()
    with pytest.raises(TypeError) as excinfo:
        _ = tree.stats["expansion_calls"]
    msg = str(excinfo.value)
    assert "tree.stats.expansion_calls" in msg
    assert "1.5.0" in msg


def test_tree_stats_subscript_on_unknown_key_raises_keyerror():
    """Unknown subscript keys fall through to plain KeyError."""
    tree = build_tree()
    with pytest.raises(KeyError):
        _ = tree.stats["does_not_exist"]
