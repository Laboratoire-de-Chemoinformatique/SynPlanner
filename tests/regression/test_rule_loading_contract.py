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
from chython.reactor.reactor import Reactor

from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions
from synplan.utils.loading import load_reaction_rules


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
    # cache.clear so other tests do not see this fixture's result
    load_reaction_rules.cache_clear()
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

    load_reaction_rules.cache_clear()
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

    load_reaction_rules.cache_clear()
    loaded = load_reaction_rules(str(legacy_pickle))
    assert len(loaded) == 2
    for i, item in enumerate(loaded):
        assert isinstance(item, Reactor), (
            f"_load_rules_pickle returned {type(item).__name__} at index {i} "
            "instead of unpacking (Reactor, priority) tuples to bare Reactors. "
            "Downstream MCTS code will fail with 'tuple object is not "
            "callable' when it tries to apply the rule."
        )
