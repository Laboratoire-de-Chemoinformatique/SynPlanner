import importlib
import sys

import pytest
from chython import smarts as smarts_parser

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction.rules.symmetry import needs_decollapsed_matches
from synplan.chem.utils import mol_from_smiles
from synplan.utils.loading import load_reaction_rules


def _needs_decollapsed_matches(rule_smarts: str) -> bool:
    return needs_decollapsed_matches(smarts_parser(rule_smarts))


def test_symmetry_helpers_are_exposed_only_from_canonical_rules_package():
    sys.modules.pop("synplan.chem.reaction_rules", None)
    sys.modules.pop("synplan.chem.reaction_rules.symmetry", None)

    with pytest.warns(DeprecationWarning):
        legacy_rules = importlib.import_module("synplan.chem.reaction_rules")

    assert not hasattr(legacy_rules, "needs_decollapsed_matches")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("synplan.chem.reaction_rules.symmetry")


def test_symmetry_module_exports_only_public_predicate():
    symmetry_module = importlib.import_module("synplan.chem.reaction.rules.symmetry")

    assert symmetry_module.__all__ == ["needs_decollapsed_matches"]


def test_useful_symmetric_rule_requires_automorphism_filter_off():
    rule_smarts = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    assert _needs_decollapsed_matches(rule_smarts)

    x_molecule = mol_from_smiles("C1=CC(C2C=CC=NC=2)=CC(C)=C1")
    reactor = CanonicalRetroReactor.from_smarts(
        rule_smarts, delete_atoms=False, automorphism_filter=False
    )

    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 2

    reactor._automorphism_filter = True
    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 1


def test_symmetric_rule_detection_does_not_leak_to_later_rules(tmp_path):
    useful_rule = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    later_non_bmg_rule = (
        "[C:1]-[C:2](-[C:3])-[N:5]-[C:4]>>[C:1]-[C:2](-[C:3])=[O:6].[C:4]-[N:5]"
    )
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        "rule_smarts\tpopularity\treaction_indices\n"
        f"{useful_rule}\t1\t0\n"
        f"{later_non_bmg_rule}\t1\t1\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    useful_reactor, later_reactor = load_reaction_rules(
        str(rules_path), decollapse_symmetric_matches=True
    )
    load_reaction_rules.cache_clear()

    assert useful_reactor._automorphism_filter is False
    assert later_reactor._automorphism_filter is not False


def test_symmetric_match_decollapsing_can_be_disabled(tmp_path):
    useful_rule = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{useful_rule}\t1\t0\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    (reactor,) = load_reaction_rules(
        str(rules_path), decollapse_symmetric_matches=False
    )
    load_reaction_rules.cache_clear()

    assert reactor._automorphism_filter is not False


def test_symmetric_rule_detection_loads_non_bmg_family(tmp_path):
    useful_non_bmg_rule = (
        "[c:2](:[c:3])(:[c:4])-[c:6](:[c:7]):[c:8]>>"
        "[I:5]-[c:6](:[c:7]):[c:8]."
        "[Cl:1]-[c:2](:[c:3]):[c:4]"
    )
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{useful_non_bmg_rule}\t1\t0\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    (reactor,) = load_reaction_rules(str(rules_path), decollapse_symmetric_matches=True)
    load_reaction_rules.cache_clear()

    assert reactor._automorphism_filter is False


def test_unlisted_external_handles_are_detected_without_chemistry_allowlist():
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[Xe:3].[C:2]-[Kr:4]"

    assert _needs_decollapsed_matches(rule_smarts)


def test_symmetric_lhs_with_equivalent_rhs_handles_is_not_decollapsed():
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[Cl:3].[C:2]-[Cl:4]"

    assert not _needs_decollapsed_matches(rule_smarts)


def test_overlapping_asymmetric_lhs_query_is_outside_formal_symmetry_scope():
    rule_smarts = "[C:1]-[C;H2,H3:2]>>[C:1]-[Cl:3].[C:2]-[Br:4]"

    assert not _needs_decollapsed_matches(rule_smarts)


def test_whole_fragment_detects_downstream_handle_difference():
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[O:3]-[C:5].[C:2]-[O:4]-[C:6]-[C:7]"

    assert _needs_decollapsed_matches(rule_smarts)


def test_loader_decollapses_downstream_handle_difference(tmp_path):
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[O:3]-[C:5].[C:2]-[O:4]-[C:6]-[C:7]"
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{rule_smarts}\t1\t0\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    (reactor,) = load_reaction_rules(str(rules_path), decollapse_symmetric_matches=True)
    load_reaction_rules.cache_clear()

    assert reactor._automorphism_filter is False


def test_full_rhs_patch_detects_target_bond_asymmetry_without_external_handles():
    rule_smarts = "[c:1]-[C:2](-[C:3])-[C:4]>>[c:1]-[C:2](-[C:3])=[C:4]"

    assert _needs_decollapsed_matches(rule_smarts)


def test_loader_decollapses_target_bond_asymmetry_without_external_handles(tmp_path):
    rule_smarts = "[c:1]-[C:2](-[C:3])-[C:4]>>[c:1]-[C:2](-[C:3])=[C:4]"
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{rule_smarts}\t1\t0\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    (reactor,) = load_reaction_rules(str(rules_path), decollapse_symmetric_matches=True)
    load_reaction_rules.cache_clear()

    assert reactor._automorphism_filter is False


def test_full_rhs_patch_keeps_equivalent_target_bond_collapsed():
    rule_smarts = "[C:1]-[C:2]>>[C:1]=[C:2]"

    assert not _needs_decollapsed_matches(rule_smarts)


def test_equivalent_branched_whole_fragments_are_not_decollapsed():
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[C:3](-[F:5])-[Cl:6].[C:2]-[C:4](-[F:7])-[Cl:8]"

    assert not _needs_decollapsed_matches(rule_smarts)


@pytest.mark.parametrize(
    ("rule_smarts", "expected"),
    [
        pytest.param(
            "[C:1]-[C:2]>>[C:1]-[C:3]1-[C:4]-[C:5]-1.[C:2]-[C:6]1-[C:7]-[C:8]-[C:9]-1",
            True,
            id="cyclopropyl-vs-cyclobutyl",
        ),
        pytest.param(
            "[C:1]-[C:2]>>[C:1]-[C:3]1-[C:4]-[C:5]-1.[C:2]-[C:6]1-[C:7]-[C:8]-1",
            False,
            id="equivalent-cyclopropyl",
        ),
    ],
)
def test_ring_whole_fragment_handles(rule_smarts, expected):
    assert _needs_decollapsed_matches(rule_smarts) is expected


def test_reconnecting_handles_are_normalized_by_lhs_automorphism():
    rule_smarts = "[C:1]-[C:2]>>[C:1]-[O:3]-[C:2].[C:2]-[O:4]-[C:1]"

    assert not _needs_decollapsed_matches(rule_smarts)


def test_product_only_atom_numbering_does_not_affect_decollapse_decision():
    low_numbers = "[C:1]-[C:2]>>[C:1]-[O:3]-[C:5].[C:2]-[Cl:7]"
    high_numbers = "[C:1]-[C:2]>>[C:1]-[O:30]-[C:50].[C:2]-[Cl:70]"

    assert _needs_decollapsed_matches(low_numbers)
    assert _needs_decollapsed_matches(high_numbers)


@pytest.mark.parametrize(
    ("metal", "metal_handle"),
    [
        pytest.param("Zn", "[Zn:4]-[Br:5]", id="zinc"),
        pytest.param("Cu", "[Cu:4]", id="copper"),
        pytest.param("Li", "[Li:4]", id="lithium"),
        pytest.param("Al", "[Al:4](-[Cl:5])-[Cl:6]", id="aluminium"),
    ],
)
def test_useful_symmetric_rule_detects_extra_organometallic_metals(metal, metal_handle):
    rule_smarts = f"[C:1]-[C:2]>>[C:1]=[O:3].[C:2]-{metal_handle}"

    assert metal in rule_smarts
    assert _needs_decollapsed_matches(rule_smarts)


@pytest.mark.parametrize(
    ("metal", "metal_handle"),
    [
        pytest.param("Zn", "[Zn:3]-[Br:5]", id="zinc"),
        pytest.param("Cu", "[Cu:3]", id="copper"),
        pytest.param("Li", "[Li:3]", id="lithium"),
        pytest.param("Al", "[Al:3](-[Cl:5])-[Cl:6]", id="aluminium"),
    ],
)
def test_generic_decollapsing_does_not_require_known_complementary_handle(
    metal, metal_handle
):
    rule_smarts = f"[C:1]-[C:2]>>[C:1]-{metal_handle}.[C:2]-[N:4]"

    assert metal in rule_smarts
    assert _needs_decollapsed_matches(rule_smarts)


@pytest.mark.parametrize(
    "rule_smarts",
    [
        pytest.param(
            "[c:1](:[c:2])(:[c:3])-[c:5](:[c:4]):[c:6]>>"
            "[c:1](:[c:2])(:[c:3])-[Br:7]."
            "[c:4]:[c:5](:[c:6])-[Sn:8](-[C:9])(-[C:10])-[C:11]",
            id="stille-like-sn",
        ),
        pytest.param(
            "[c:2](:[c:3])(:[c:4])-[c:6](:[c:7]):[c:8]>>"
            "[I:5]-[c:6](:[c:7]):[c:8]."
            "[Cl:1]-[c:2](:[c:3]):[c:4]",
            id="mixed-halogen-coupling",
        ),
        pytest.param(
            "[C:1](-[C:2])=[C:4]-[C:3]>>"
            "[C:1](-[C:2])=[O:5]."
            "[C:3]-[C:4]=[P:6](-[c:7]:1:[c:8]:[c:9]:[c:10]:"
            "[c:11]:[c:12]:1)(-[c:13]:2:[c:14]:[c:15]:[c:16]:"
            "[c:17]:[c:18]:2)-[c:19]:3:[c:20]:[c:21]:[c:22]:"
            "[c:23]:[c:24]:3",
            id="wittig-like",
        ),
        pytest.param(
            "[C:1]=[C:2]>>"
            "[C:1]=[O:3]."
            "[C:2]-[S:4](=[O:5])(=[O:6])-[c:7]:1:[n:8]:"
            "[n:9]:[c:10]:[s:11]:1",
            id="julia-kocienski-like",
        ),
        pytest.param(
            "[C:1]=[C:2]>>[C:1]=[O:3].[C:2]-[Si:4](-[C:5])(-[C:6])-[C:7]",
            id="peterson-like",
        ),
        pytest.param(
            "[C:1]-[C:2]>>[C:1]=[O:3].[C:2]-[Cl:4]",
            id="benzaldehyde-benzyl-chloride",
        ),
        pytest.param(
            "[c:1]-[N:2]=[N:3]-[c:4]>>[c:1]-[N+:5]#[N:6].[c:4]-[H:7]",
            id="diazotisation-coupling",
        ),
    ],
)
def test_useful_symmetric_rule_detects_expanded_precursor_families(rule_smarts):
    assert _needs_decollapsed_matches(rule_smarts)


@pytest.mark.parametrize(
    "rule_smarts",
    [
        pytest.param(
            "[C:1]-[C:2]=[C:5]-[C:6]>>[C:4]=[C:5]-[C:6].[C:1]-[C:2]=[C:3]",
            id="cross-metathesis-equivalent-handles",
        ),
        pytest.param(
            "[c:1]-[C:2]=[C:3]-[c:4]>>"
            "[Br:5]-[c:1].[C:2](-[Br:6])=[C:3](-[Br:7]).[Br:8]-[c:4]",
            id="dibromoalkene-equivalent-handles",
        ),
        pytest.param(
            "[n:1]:1:[c:2](:[c:3]):[c:5](:[c:4]):[n:6]:[c:8]:1-[C:7]>>"
            "[N:1]-[c:2](:[c:3]):[c:5](:[c:4])-[N:6]."
            "[C:7]-[C:8]#[N:9]",
            id="nitrile-handle-on-automorphism-fixed-atom",
        ),
    ],
)
def test_universal_predicate_ignores_equivalent_or_fixed_handles(rule_smarts):
    assert not _needs_decollapsed_matches(rule_smarts)


@pytest.mark.parametrize(
    "rule_smarts",
    [
        pytest.param(
            "[c:2](:[c:3])(:[c:4])-[B:7](-[O:8])-[O:12]>>"
            "[Br:1]-[c:2](:[c:3]):[c:4]."
            "[C:5](-[O:6]-[B:7](-[O:8]-[C:9](-[C:10])-[C:11])"
            "-[O:12]-[C:13](-[C:14])-[C:15])(-[C:16])-[C:17]",
            id="boronate-ester-modification",
        ),
        pytest.param(
            "[C:1]-[C:3]-1(-[C:2])-[O:9]-[C:10]-[C:11]-[O:12]-1>>"
            "[C:5]-[Si:6](-[C:7])(-[C:8])-[O:9]-[C:10]-[C:11]"
            "-[O:12]-[Si:13](-[C:14])(-[C:15])-[C:16]."
            "[C:1]-[C:3](-[C:2])=[O:4]",
            id="silyl-protection",
        ),
        pytest.param(
            "[C:1]-[C:2](-[C:3])-[N:5]-[C:4]>>[C:1]-[C:2](-[C:3])=[O:6].[C:4]-[N:5]",
            id="amide-noncoupling",
        ),
    ],
)
def test_symmetric_rule_detection_excludes_nonuseful_precursors(rule_smarts):
    assert not _needs_decollapsed_matches(rule_smarts)
