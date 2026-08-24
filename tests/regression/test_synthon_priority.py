"""The synthon disconnections must reach the tree and fire, not merely parse.

Their labels are inert on the reactor path unless the leaving group is spelled out, so every
assertion here is written to fail loudly if capping, the priority wiring, or the fragment-count
prior quietly stops doing anything.
"""

from __future__ import annotations

import re
from unittest import mock

import pytest
from chython import smarts, smiles, synthon_smiles

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.reactor import apply_reaction_rule
from synplan.chem.reaction.rules import (
    POLICY_SOURCE_NAME,
    parse_priority_rules,
    rule_query_pattern,
)
from synplan.chem.reaction.rules import synthon as priority
from synplan.chem.reaction.rules.synthon import (
    SYNTHON_SOURCE_NAME,
    _records,
    capped_smarts,
    synthon_priority_rules,
)
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.transformer import SynthonTransformer, query_labels
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RandomEvaluationStrategy
from synplan.mcts.policy.base import Policy
from synplan.mcts.tree import Tree
from synplan.utils.visualisation import (
    _format_arrow_label,
    get_route_svg,
    route_rule_labels,
)

AMIDE = "CCNC(=O)c1ccccc1"
AMISULPRIDE = "CCCN1CCCC1CNC(=O)c1cc(S(C)(=O)=O)c(N)cc1OC"
PANEL = (AMIDE, AMISULPRIDE, "CC(O)c1ccccc1", "c1ccccc1-c1ccncc1", "CCOC(=O)c1ccccc1")
STUB_POLICY_PROB = 0.05


class _StubPolicy(Policy):
    """A policy-shaped probability over a fixed rule list; no network, no downloads."""

    config = None

    def __init__(self, rules) -> None:
        self.rules = tuple(rules)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def predict_reaction_rules(self, precursor, reaction_rules):
        for rule_id, rule in enumerate(self.rules):
            yield STUB_POLICY_PROB, rule, rule_id


def _molecule(smi: str):
    mol = smiles(smi)
    mol.canonicalize()
    return mol


def _rule(rule_id: str, **kwargs) -> CanonicalRetroReactor:
    records = [
        r
        for r in load_data(SynthonConfig().rules_path)["disconnections"]
        if not r["macro"]
    ]
    index = next(i for i, r in enumerate(records) if r["id"] == rule_id)
    return synthon_priority_rules(**kwargs)[SYNTHON_SOURCE_NAME][index]


def _fired(cap: bool) -> set[int]:
    rules = synthon_priority_rules(cap=cap)[SYNTHON_SOURCE_NAME]
    fired: set[int] = set()
    for smi in PANEL:
        mol = _molecule(smi)
        for index, rule in enumerate(rules):
            for products in apply_reaction_rule(mol, rule):
                if products:
                    fired.add(index)
                    break
    return fired


@pytest.fixture(scope="module")
def priority_tree() -> Tree:
    """A real search with the real priority path; only the neural policy is stubbed.

    The stub ranks the H-capped variants of the same disconnections: they fire on the same
    target and yield different products, so the tree really does get policy siblings to compare
    the priority prior against.
    """
    policy = _StubPolicy(synthon_priority_rules(cap=False)[SYNTHON_SOURCE_NAME])
    tree = Tree(
        target=_molecule(AMISULPRIDE),
        config=TreeConfig(
            use_priority=True,
            max_iterations=3,
            search_strategy="expansion_first",
            silent=True,
        ),
        reaction_rules=policy.rules,
        building_blocks={"CCN"},
        expansion_function=policy,
        evaluation_function=RandomEvaluationStrategy(),
        priority_rules=synthon_priority_rules(),
    )
    for _ in tree:
        pass
    return tree


def test_loader_returns_reactors_under_one_non_reserved_name() -> None:
    rules = synthon_priority_rules()
    assert list(rules) == [SYNTHON_SOURCE_NAME]
    assert SYNTHON_SOURCE_NAME != POLICY_SOURCE_NAME
    assert len(rules[SYNTHON_SOURCE_NAME]) == 108  # 39 acyclic + the authored ring rules
    assert len(synthon_priority_rules(macro=True)[SYNTHON_SOURCE_NAME]) == 147
    assert all(
        isinstance(rule, CanonicalRetroReactor) for rule in rules[SYNTHON_SOURCE_NAME]
    )


def test_the_reserved_policy_name_is_refused() -> None:
    with pytest.raises(ValueError, match="reserved"):
        parse_priority_rules({POLICY_SOURCE_NAME: ["[C:1]-!@[C:2]>>[C:1].[C:2]"]})


def test_every_rule_carries_a_query_pattern() -> None:
    """`PriorityPolicy` gates a pattern-less rule off forever, without raising."""
    for rule in synthon_priority_rules()[SYNTHON_SOURCE_NAME]:
        assert rule_query_pattern(rule) is not None


def test_capping_gives_the_acyl_chloride_not_the_aldehyde() -> None:
    """The loaded default must be the capped set; the H-capped one is chemically inert."""
    target = _molecule(AMIDE)
    capped = [
        sorted(str(p) for p in products)
        for products in apply_reaction_rule(target, _rule("R1.1"))
    ]
    assert capped == [["CCN", "c1ccccc1C(Cl)=O"]]
    inert = [
        sorted(str(p) for p in products)
        for products in apply_reaction_rule(target, _rule("R1.1", cap=False))
    ]
    assert inert == [["CCN", "c1ccccc1C=O"]]


def test_the_raw_rules_carry_labels_and_the_capped_ones_carry_none() -> None:
    """Capping trades every token for a spelled-out leaving group — 78 labels in, 0 out.

    Both halves matter. If the raw set stops carrying 78, a rule lost its labels upstream in
    ``rules.json`` and the disconnection now proposes the wrong reagent class. If the capped set
    ever carries one, some caller is about to trust a label that ``str(Reactor)`` will drop.
    The ring rules are out of scope: they never see the capper, they ship a reagent form.
    """
    config = SynthonConfig()
    leaving_groups = load_data(config.rules_path)["leaving_groups"]
    records = [r for r in _records(config, macro=False) if not r["ring"]]

    def labels(rule_smarts: str) -> int:
        return len(query_labels(smarts(rule_smarts.split(">>", 1)[1].strip())))

    assert sum(labels(r["smarts"]) for r in records) == 78
    assert (
        sum(
            labels(capped_smarts(r["smarts"], leaving_groups, r["id"])) for r in records
        )
        == 0
    )


def test_a_token_the_capper_cannot_spell_is_loud() -> None:
    """A ninth token, or any spec ``_LABELLED`` misses, must raise instead of passing through.

    Simulated by narrowing the regex rather than by inventing a label: the eight are fixed, and
    the failure being guarded against is precisely the parser and the regex drifting apart.
    """
    record = next(
        r for r in _records(SynthonConfig(), macro=False) if r["id"] == "R1.1"
    )
    leaving_groups = load_data(SynthonConfig().rules_path)["leaving_groups"]
    with (
        mock.patch.object(priority, "_LABELLED", re.compile(r"(?!x)x()()()")),
        pytest.raises(ValueError, match=r"R1\.1: labelled RHS atoms .* kept their"),
    ):
        capped_smarts(record["smarts"], leaving_groups, record["id"])


def test_capping_never_silently_deletes_a_rule() -> None:
    """Capping `C:nuc` with the shipped bare `[Mg]` kills six rules and logs nothing."""
    assert _fired(cap=True) >= _fired(cap=False)


def test_priority_rules_reach_the_tree_and_fire(priority_tree: Tree) -> None:
    sources = {
        node.rule_source for node in priority_tree.nodes.values() if node.rule_source
    }
    assert SYNTHON_SOURCE_NAME in sources
    assert POLICY_SOURCE_NAME in sources


def test_per_source_stats_are_recorded(priority_tree: Tree) -> None:
    counters = priority_tree.stats.per_priority_source[SYNTHON_SOURCE_NAME]
    assert counters.tried > 0
    assert counters.succeeded > 0


def test_the_fragment_count_prior_outranks_every_policy_sibling(
    priority_tree: Tree,
) -> None:
    siblings = priority_tree.children[1]
    synthon = [
        priority_tree.nodes[i].prob
        for i in siblings
        if priority_tree.nodes[i].rule_source == SYNTHON_SOURCE_NAME
    ]
    policy = [
        priority_tree.nodes[i].prob
        for i in siblings
        if priority_tree.nodes[i].rule_source == POLICY_SOURCE_NAME
    ]
    assert policy, "no policy siblings — the comparison would be vacuous"
    assert max(policy) <= 2 * STUB_POLICY_PROB
    assert max(synthon) >= 2.0
    assert max(synthon) > max(policy)


def test_the_cap_is_pinned_atom_for_atom() -> None:
    """Aromaticity and bond order are read off the rule, and nothing else proves they were.

    Both branches survive being deleted: the whole suite stays green with the aromatic lookup
    keyed on `C` instead of `c` (9 rules swap reagent class — R12.1's Suzuki turns from ArBr to
    ArCl) and with the `elec2/nuc2/neut2` bond order forced to single (R11.2's alkene turns into
    an alkane). Only the exact string catches either.
    """
    data = load_data(SynthonConfig().rules_path)
    leaving_groups = data["leaving_groups"]
    by_id = {r["id"]: r["smarts"] for r in data["disconnections"] if not r["macro"]}

    def rhs(rule_id: str) -> str:
        return capped_smarts(by_id[rule_id], leaving_groups).split(">>")[1]

    assert rhs("R12.1") == "[c:1]-[Br:90].[c:2]-[B:91]([O:92])[O:93]"
    assert rhs("R3.2") == "[N:2].[#6:1]-[Br:90]"
    assert rhs("R11.2") == "[#6;+0:1]=[C:90].[#6;+0:2]=[C:91]"
    assert rhs("R9.1") == "[#6:1]=[O:90].[N:2]"


def test_a_transformer_exposes_a_query_pattern() -> None:
    """`Reactor` keeps its patterns on `_patterns`, `Transformer` on `_pattern`.

    Without the singular fallback `PriorityPolicy._rule_applies` returns False forever and a
    transformer-based priority set is gated off with no error — inert, not wrong.
    """
    data = load_data(SynthonConfig().rules_path)
    smarts_text = next(r["smarts"] for r in data["disconnections"] if r["id"] == "R1.1")
    assert rule_query_pattern(SynthonTransformer.from_smarts(smarts_text)) is not None


def test_a_label_blind_rule_refuses_a_labelled_synthon() -> None:
    """`QueryElement.__eq__` never consults `_label`, so a plain reactor matches a labelled
    synthon and emits unlabelled products. The labels vanish with no error unless this raises."""
    labelled = synthon_smiles("[NH2_nuc]CC")
    labelled.canonicalize()
    rule = synthon_priority_rules()[SYNTHON_SOURCE_NAME][0]
    with pytest.raises(TypeError, match="silently strip the labels"):
        list(apply_reaction_rule(labelled, rule))


def test_a_route_report_names_the_reaction_a_priority_rule_applied(
    priority_tree: Tree,
) -> None:
    """The tree stores only `(rule_source, rule_id)`, so the chemistry name has to travel on the
    rule object. Without it a report shows a SMILES step and no clue which reaction was applied.
    """
    rules = synthon_priority_rules()[SYNTHON_SOURCE_NAME]
    assert all(getattr(r, "rule_name", None) for r in rules)
    assert rules[0].rule_id.startswith("R")

    synthon_child = next(
        i
        for i in priority_tree.children[1]
        if priority_tree.nodes[i].rule_source == SYNTHON_SOURCE_NAME
    )
    labels = route_rule_labels(priority_tree, synthon_child)
    assert any("—" in label for label in labels), labels
    named = next(label for label in labels if "—" in label)
    rule = rules[priority_tree.nodes[synthon_child].rule_id]
    assert named == f"{rule.rule_id} — {rule.rule_name}"


def test_a_carbon_nucleophile_caps_to_a_real_reagent() -> None:
    """`rules.json` writes a carbon nucleophile as bare `[Mg]`, which is shorthand. Falling back
    to H proposes an ALKANE — purchasable, so the route terminates, but unable to do the reaction.

    The negative matters as much as the positives: a Sonogashira nucleophile really is the
    terminal alkyne, so R12.4 must keep H and must not acquire a Grignard.
    """
    data = load_data(SynthonConfig().rules_path)
    leaving_groups = data["leaving_groups"]
    by_id = {r["id"]: r["smarts"] for r in data["disconnections"] if not r["macro"]}

    def rhs(rule_id: str) -> str:
        return capped_smarts(by_id[rule_id], leaving_groups, rule_id).split(">>")[1]

    assert rhs("R10.1") == "[#6:1]-[Cl:90].[#6:2]-[Mg:91][Br:92]"
    assert rhs("R12.5") == "[#6:1]-[Br:90].[#6:2]-[Mg:91][Br:92]"
    assert rhs("R12.3b") == "[c:1]-[B:90]([O:91])[O:92].[C:2]-[Cl:93]"
    assert rhs("R12.4") == "[c:1]-[Br:90].[C:2]"

    # and no rule may still be carrying the shorthand through to a reactor
    for rule_id, smarts_text in by_id.items():
        assert "[Mg]." not in capped_smarts(smarts_text, leaving_groups, rule_id)


def test_the_route_svg_labels_its_arrows_with_the_reaction_name(
    priority_tree: Tree,
) -> None:
    """The name belongs on the arrow, not only in the step list underneath — that is where a
    reader looks to see which reaction a disconnection stands for.
    """
    synthon_child = next(
        i
        for i in priority_tree.children[1]
        if priority_tree.nodes[i].rule_source == SYNTHON_SOURCE_NAME
    )
    svg = get_route_svg(priority_tree, synthon_child, allow_unsolved=True)
    rule = synthon_priority_rules()[SYNTHON_SOURCE_NAME][
        priority_tree.nodes[synthon_child].rule_id
    ]
    assert rule.rule_id in svg
    assert "<text" in svg

    # a long name is truncated rather than allowed to run into the neighbouring molecule
    long_name = _format_arrow_label(
        None, None, include_rule_key=False, rule_name="R10.1 " + "x" * 80
    )
    assert len(long_name) == 25 and long_name.endswith("…")
