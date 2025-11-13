import random
import time
from typing import Callable, List, Tuple

from CGRtools.containers import MoleculeContainer

from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig, RolloutEvaluationConfig
from synplan.utils.loading import load_evaluation_function
from synplan.mcts.algorithm import NestedMonteCarlo


# ----------------------
# Helper constructors
# ----------------------


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
    def __init__(
        self, reactants: List[MoleculeContainer], products: List[MoleculeContainer]
    ):
        self.reactants = reactants
        self.products = products


class FakeReactor:
    def __init__(self, products_fn: Callable[[], List[MoleculeContainer]]):
        self.products_fn = products_fn

    def __call__(self, reactants: List[MoleculeContainer]):
        return [FakeReaction(reactants, self.products_fn())]


class FakePolicy:
    def __init__(
        self, rules: List[Tuple[float, FakeReactor, int]], expand_deeper: bool = False
    ):
        # rules: list of (prob, reactor, rule_id)
        self.rules = rules
        self.expand_deeper = expand_deeper

    def predict_reaction_rules(self, precursor, reaction_rules):
        # only generate at the root by default (len(prev_precursors) == 1 at root)
        if not self.expand_deeper and len(precursor.prev_precursors) > 1:
            return
        for prob, reactor, rid in self.rules:
            yield prob, reactor, rid


def build_tree(
    algorithm: str,
    rules: List[Tuple[float, FakeReactor, int]],
    *,
    min_mol_size: int = 6,
    beam_width: int = 1,
    ucb_type: str = "puct",
    epsilon: float = 0.0,
    expand_deeper: bool = False,
) -> Tree:
    cfg = TreeConfig(
        algorithm=algorithm,
        max_iterations=5,
        max_tree_size=100,
        max_time=10,
        max_depth=3,
        search_strategy="expansion_first",
        beam_width=beam_width,
        ucb_type=ucb_type,
        epsilon=epsilon,
        min_mol_size=min_mol_size,
        silent=True,
        enable_pruning=False,
    )
    target = make_mol(7)  # ensure not a building block
    fake_policy = FakePolicy(rules, expand_deeper=expand_deeper)
    reactors = [r for _, r, _ in rules]
    evaluation_config = RolloutEvaluationConfig(
        policy_network=fake_policy, reaction_rules=reactors, building_blocks=set()
    )
    evaluator = load_evaluation_function(evaluation_config)
    return Tree(
        target=target,
        config=cfg,
        reaction_rules=reactors,
        building_blocks=set(),
        expansion_function=fake_policy,
        evaluation_function=evaluator,
    )


# ----------------------
# Breadth-First Search
# ----------------------


def test_breadth_first_returns_solved_child():
    # one rule yields building-block product (solved child), another yields large product (unsolved)
    rules = [
        (0.5, FakeReactor(lambda: [make_mol(5)]), 0),  # solved child (<= min_mol_size)
        (0.5, FakeReactor(lambda: [make_mol(10)]), 1),  # unsolved child
    ]
    tree = build_tree("breadth_first", rules)

    found, node_ids = tree.algorithm.step()
    assert found is True
    child_id = node_ids[0]
    assert tree.nodes[child_id].is_solved() is True
    assert tree.found_a_route is True
    assert len(tree.winning_nodes) >= 1


# ----------------------
# Best-First Search
# ----------------------


def test_best_first_orders_by_policy_value():
    # two unsolved children with different policy priors
    rules = [
        (0.1, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.9, FakeReactor(lambda: [make_mol(12)]), 1),
    ]
    tree = build_tree("best_first", rules)

    found, node_ids = tree.algorithm.step()
    assert found is False
    # frontier should be sorted by nodes_prob (policy value)
    assert len(tree.algorithm.frontier) == 2
    # frontier is sorted by evaluation score used for insertion
    best_score = max(tree._get_node_value(cid) for cid in tree.children[1])
    assert tree.algorithm.frontier[0][1] == best_score


def test_best_first_returns_solved_child_immediately():
    rules = [
        (0.2, FakeReactor(lambda: [make_mol(5)]), 0),  # solved
        (0.8, FakeReactor(lambda: [make_mol(10)]), 1),
    ]
    tree = build_tree("best_first", rules)

    found, node_ids = tree.algorithm.step()
    assert found is True
    assert tree.nodes[node_ids[0]].is_solved() is True
    assert tree.found_a_route is True


# ----------------------
# Beam Search
# ----------------------


def test_beam_top1_is_highest():
    rules = [
        (0.3, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.7, FakeReactor(lambda: [make_mol(12)]), 1),
    ]
    tree = build_tree("beam", rules, beam_width=1)

    found, node_ids = tree.algorithm.step()
    assert found is False
    assert len(tree.algorithm.frontier) == 2
    best_score = max(tree._get_node_value(cid) for cid in tree.children[1])
    assert tree.algorithm.frontier[0][1] == best_score


# ----------------------
# UCT/PUCT
# ----------------------


def test_uct_puct_selects_highest_prior_on_second_step():
    rules = [
        (0.4, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.9, FakeReactor(lambda: [make_mol(12)]), 1),
    ]
    tree = build_tree("uct", rules, ucb_type="puct", epsilon=0.0)

    # step 1: expand root
    found, ids = tree.algorithm.step()
    assert found is False
    assert ids == [1]

    # step 2: should select the highest-prior child
    found2, ids2 = tree.algorithm.step()
    assert found2 in (False, True)
    selected_id = ids2[0]
    best_id = max(list(tree.children[1]), key=lambda cid: tree.nodes_prob[cid])
    assert selected_id == best_id


# ----------------------
# Nested Monte Carlo (NMCS)
# ----------------------


def test_nmcs_returns_best_leaf_id():
    rules = [
        (0.2, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.8, FakeReactor(lambda: [make_mol(12)]), 1),
    ]
    # disable deeper expansions so playout evaluation equals child's policy value
    tree = build_tree("nmcs", rules, expand_deeper=False)

    # ensure children exist and time budget active before NMCS step
    tree._expand_node(1)
    tree.expanded_nodes.add(1)
    tree.start_time = time.time()
    found, ids = tree.algorithm.step()
    assert found in (False, True)
    assert ids[0] in tree.children[1]
    best_score = max(tree._get_node_value(cid) for cid in tree.children[1])
    assert tree._get_node_value(ids[0]) == best_score


def test_nmcs_marks_solved_when_present():
    rules = [
        (0.9, FakeReactor(lambda: [make_mol(5)]), 0),  # solved child
        (0.1, FakeReactor(lambda: [make_mol(10)]), 1),
    ]
    tree = build_tree("nmcs", rules, expand_deeper=False)

    # pre-expand to expose solved child and start timing
    tree._expand_node(1)
    tree.expanded_nodes.add(1)
    tree.start_time = time.time()
    found, _ = tree.algorithm.step()
    # NMCS should mark solved children
    assert any(tree.nodes[cid].is_solved() for cid in tree.children[1])
    assert tree.found_a_route is True
    assert len(tree.winning_nodes) >= 1


# ----------------------
# Lazy NMCS
# ----------------------


def test_lazy_nmcs_selects_best_candidate_after_pruning():
    rules = [
        (0.1, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.5, FakeReactor(lambda: [make_mol(11)]), 1),
        (0.9, FakeReactor(lambda: [make_mol(12)]), 2),
    ]
    tree = build_tree("lazy_nmcs", rules, expand_deeper=False)

    # pre-expand root and start timing
    tree._expand_node(1)
    tree.expanded_nodes.add(1)
    tree.start_time = time.time()
    found, ids = tree.algorithm.step()
    assert found in (False, True)
    assert ids[0] in tree.children[1]
    best_score = max(tree._get_node_value(cid) for cid in tree.children[1])
    assert tree._get_node_value(ids[0]) == best_score


# ----------------------
# NMCS playout helper on real tree
# ----------------------


def test_select_nmcs_path_greedy_policy_and_random_on_real_tree():
    rules = [
        (0.2, FakeReactor(lambda: [make_mol(10)]), 0),
        (0.8, FakeReactor(lambda: [make_mol(12)]), 1),
    ]
    tree = build_tree("best_first", rules, expand_deeper=False)
    # pre-expand to ensure children present and start timing
    tree._expand_node(1)
    tree.expanded_nodes.add(1)
    tree.start_time = time.time()
    playout = NestedMonteCarlo(tree)

    # greedy should follow highest value (policy value here)
    last_greedy, seq_g = playout.select_nmcs_path(1, 1, "greedy")
    assert len(seq_g) == 1
    best_by_value = max(
        list(tree.children[1]), key=lambda cid: tree._get_node_value(cid)
    )
    assert seq_g[0] == best_by_value
    assert last_greedy in list(tree.children[1])

    # policy should follow highest prior as well
    last_pol, seq_p = playout.select_nmcs_path(1, 1, "policy")
    assert len(seq_p) == 1
    best_prior = max(list(tree.children[1]), key=lambda cid: tree.nodes_prob[cid])
    assert seq_p[0] == best_prior
    assert last_pol in list(tree.children[1])

    # random should still return a valid child deterministically seeded
    random.seed(0)
    last_rnd, seq_r = playout.select_nmcs_path(1, 1, "random")
    assert len(seq_r) == 1
    assert seq_r[0] in list(tree.children[1])
    assert last_rnd in list(tree.children[1])
