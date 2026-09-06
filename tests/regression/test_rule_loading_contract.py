"""Contract tests for ``load_reaction_rules`` and its helpers.

The public function declares ``-> tuple[Reactor, ...]`` and is decorated
with ``functools.cache``. Returning a tuple makes the cached value
immutable, so callers (including parallel tree workers sharing the same
rules) cannot accidentally ``append``/``sort``/``extend`` and mutate the
shared state.

Separately, ``_load_rules_pickle`` is supposed to unpack the legacy
``[(Reactor, priority), ...]`` pickle format into bare Reactors, but the
isinstance check is inverted, so the unpack never runs and downstream code
gets ``(Reactor, priority)`` tuples it then tries to call as Reactors.

These tests assert the public *contract* of each entry point, not its
internal mechanics.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from chython import smiles as smiles_parser
from chython.reactor.reactor import Reactor

from synplan.chem.reaction import apply_reaction_rule
from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions
from synplan.utils.loading import load_reaction_rules

_SHIPPED_COMPONENT_LOCAL_CX_RULES = (
    "[c:1]-[C:2](-[O:3])=[O:8]>>[c:1]-[C:2]-[O:3]."
    "[C:4]-[C:5]-1(-[C:6])-[N:7](-[O:8])-[C:9](-[C:10])"
    "(-[C:11])-[C:12]-[C:13]-[C:14]-1 |^1:4|",
    "[C:1]-[C:2](=[O:3])-[O:8]>>[C:1]-[C:2]-[O:3]."
    "[C:4]-[C:5]-1(-[C:6])-[N:7](-[O:8])-[C:9](-[C:10])"
    "(-[C:11])-[C:12]-[C:13]-[C:14]-1 |^1:4|",
)


@pytest.fixture(autouse=True)
def clear_rule_cache():
    load_reaction_rules.cache_clear()
    yield
    load_reaction_rules.cache_clear()


def _write_rules_tsv(path: Path, rules: tuple[str, ...]) -> None:
    rows = ["rule_smarts\tpopularity\treaction_indices"]
    rows.extend(f"{rule}\t1\t{index}" for index, rule in enumerate(rules))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def real_rules_tsv(tmp_path: Path, sample_reactions_file, rule_cfg_factory) -> Path:
    """Run extraction once and return the resulting rules TSV."""
    rules_path = tmp_path / "rules.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=True, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_path),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    if not rules_path.exists() or rules_path.stat().st_size == 0:
        pytest.skip("extraction produced no rules; cannot exercise the loader")
    return rules_path


def test_load_reaction_rules_returns_immutable_tuple(real_rules_tsv: Path):
    """The contract is ``-> tuple[Reactor, ...]``: a sequence that supports
    indexing and ``len()`` but rejects mutation.

    The decorator ``functools.cache`` returns the same object on every call
    with the same arguments. Returning a tuple guarantees that parallel
    consumers (multiple ``Tree`` instances, training workers) cannot
    accidentally mutate the shared rule set.
    """
    rules = load_reaction_rules(str(real_rules_tsv))
    assert len(rules) > 0
    assert isinstance(rules[0], Reactor)
    assert isinstance(rules, tuple), (
        f"load_reaction_rules is annotated -> tuple[Reactor, ...] but "
        f"returned {type(rules).__name__}. The tuple is part of the "
        "contract: callers share the cached value across parallel tree "
        "workers and must not be able to mutate it."
    )


def test_load_reaction_rules_bad_smarts_message_is_actionable(
    tmp_path: Path,
    real_rules_tsv: Path,
):
    """When a SMARTS row fails to parse, the exception identifies *which* row.

    Without row context, the user has to grep a chython traceback against a
    rules file that may have thousands of rows. ``parse_priority_rules``
    already does this; ``_load_rules_tsv`` should match. We do not pin the
    exact wording, only require that the row number or the offending SMARTS
    string appears somewhere in the chain of exception messages.
    """
    bad_tsv = tmp_path / "rules_with_bad.tsv"
    rows = real_rules_tsv.read_text(encoding="utf-8").splitlines()
    bad_smarts = "this_is_not_a_valid_smarts_pattern_at_all"
    # Inject a broken row after the header + one good row.
    if len(rows) < 2:
        pytest.skip("need at least one valid rule row to inject after")
    rows.insert(2, f"{bad_smarts}\t1\t0")
    bad_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        load_reaction_rules(str(bad_tsv))
    # Walk the exception chain looking for the SMARTS or a row reference.
    messages = []
    e: BaseException | None = exc_info.value
    while e is not None:
        messages.append(str(e))
        e = e.__cause__ or e.__context__
    combined = " | ".join(messages)
    assert (
        bad_smarts in combined
        or "row" in combined.lower()
        or "line" in combined.lower()
    ), (
        "Exception from load_reaction_rules does not name the offending row "
        "or SMARTS text. Got chain:\n  " + combined + "\n"
        "Diagnosing rule files of thousands of rows requires either the row "
        "number or the SMARTS text itself in the message."
    )


def test_load_reaction_rules_legacy_pickle_unpacks_priority_tuples(tmp_path: Path):
    """Legacy ``[(Reactor, priority)]`` pickles must load as bare Reactors.

    The ``_load_rules_pickle`` code path is supposed to detect this legacy
    format and unpack it. With the isinstance check inverted, the unpack
    never runs and downstream callers see ``(Reactor, priority)`` tuples
    where they expect Reactors, producing ``TypeError: 'tuple' object is
    not callable`` deep inside MCTS.
    """
    # Build a minimal Reactor from a SMARTS that we know works.
    rxn_smarts = "[C:1][O:2]>>[C:1].[O:2]"
    try:
        reactor = Reactor.from_smarts(rxn_smarts)
    except Exception as e:  # pragma: no cover - guard for chython API drift
        pytest.skip(f"unable to build a test Reactor: {e}")

    legacy_pickle = tmp_path / "legacy_rules.pickle"
    legacy_rules = [(reactor, 5), (reactor, 3)]  # (Reactor, priority) tuples
    with open(legacy_pickle, "wb") as f:
        pickle.dump(legacy_rules, f)

    loaded = load_reaction_rules(str(legacy_pickle))
    assert len(loaded) == 2
    for i, item in enumerate(loaded):
        assert isinstance(item, Reactor), (
            f"_load_rules_pickle returned {type(item).__name__} at index {i} "
            "instead of unpacking (Reactor, priority) tuples to bare Reactors. "
            "Downstream MCTS code will fail with 'tuple object is not "
            "callable' when it tries to apply the rule."
        )


@pytest.mark.parametrize(
    ("rule_smarts", "target_smiles"),
    zip(
        _SHIPPED_COMPONENT_LOCAL_CX_RULES,
        ("O=C(O)c1ccccc1", "CC(=O)O"),
        strict=True,
    ),
    ids=["benzoic-acid", "acetic-acid"],
)
def test_shipped_tempo_rule_keeps_radical_on_tempo_oxygen(
    tmp_path: Path, rule_smarts: str, target_smiles: str
):
    """Both shipped variants must return an alcohol plus the TEMPO radical.

    The aromatic rule is data row 3555 of ``supervised_gps/v1/reaction_rules.tsv``
    (16 source reactions), copied verbatim. Both CXSMARTS indices are local to
    the second product: atom 4 in that component is the TEMPO oxygen, map 8.
    Whole-reaction parsing instead makes a carbon in the first product radical.
    """
    rules_path = tmp_path / "tempo_oxidation_rule.tsv"
    _write_rules_tsv(rules_path, (rule_smarts,))

    (reactor,) = load_reaction_rules(str(rules_path))
    assert str(reactor) == rule_smarts

    pattern_radicals = [
        (number, atom.atomic_symbol)
        for pattern in reactor._patterns
        for number, atom in pattern.atoms()
        if atom.is_radical
    ]
    product_radicals = [
        (number, atom.atomic_symbol)
        for product in reactor._products
        for number, atom in product.atoms()
        if atom.is_radical
    ]

    assert pattern_radicals == []
    assert product_radicals == [(8, "O")]

    precursor_sets = list(
        apply_reaction_rule(smiles_parser(target_smiles), reactor, top_reactions_num=10)
    )
    assert len(precursor_sets) == 1
    assert sorted(
        tuple(atom.atomic_symbol for _, atom in precursor.atoms() if atom.is_radical)
        for precursor in precursor_sets[0]
    ) == [(), ("O",)]


def test_loader_accepts_side_global_cx_radical_block(tmp_path: Path):
    """Fall back to whole-reaction parsing for side-global CX atom indices."""
    side_global_rule = "[C;D1:1].[C;D1:2] |^1:1|>>[C:1]-[C:2]"
    rules_path = tmp_path / "side_global_cx_rule.tsv"
    _write_rules_tsv(rules_path, (side_global_rule,))

    (loaded,) = load_reaction_rules(str(rules_path))

    assert str(loaded) == "[C;D1:1].[C;D1:2] |^1:0|>>[C:1]-[C:2]"


def test_loader_assigns_reaction_wide_numbers_to_unmapped_atoms(tmp_path: Path):
    """Bare atoms must not collide with explicit maps in other components."""
    partially_mapped_rule = "[O].[C:1]>>[N]-[C:1]"
    rules_path = tmp_path / "partially_mapped_rule.tsv"
    _write_rules_tsv(rules_path, (partially_mapped_rule,))

    (loaded,) = load_reaction_rules(str(rules_path))

    assert tuple(tuple(query.atoms_numbers) for query in loaded._patterns) == (
        (2,),
        (1,),
    )
    assert tuple(tuple(query.atoms_numbers) for query in loaded._products) == ((3, 1),)


def test_loader_numbers_unmapped_atoms_after_reaction_cx_fallback(tmp_path: Path):
    """Whole-reaction CX parsing must use the same global number allocator."""
    partially_mapped_rule = "[O].[C:1] |^1:1|>>[N]-[C:1]"
    rules_path = tmp_path / "partially_mapped_cx_rule.tsv"
    _write_rules_tsv(rules_path, (partially_mapped_rule,))

    (loaded,) = load_reaction_rules(str(rules_path))

    assert tuple(tuple(query.atoms_numbers) for query in loaded._patterns) == (
        (2,),
        (1,),
    )
    assert tuple(tuple(query.atoms_numbers) for query in loaded._products) == ((3, 1),)


def test_loader_rejects_reaction_smarts_with_reagents(tmp_path: Path):
    """The component parser must preserve the loader's reagent rejection."""
    rules_path = tmp_path / "rule_with_reagent.tsv"
    _write_rules_tsv(rules_path, ("[C:1]>[O]>[C:1]",))

    with pytest.raises(ValueError, match="reagents are not supported"):
        load_reaction_rules(str(rules_path))
