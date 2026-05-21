import pytest

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction_rules.symmetry import is_useful_symmetric_reaction_rule
from synplan.chem.utils import mol_from_smiles
from synplan.utils.loading import load_reaction_rules


def test_useful_symmetric_rule_requires_automorphism_filter_off():
    rule_smarts = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    assert is_useful_symmetric_reaction_rule(rule_smarts)

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
        str(rules_path), detect_symmetric_rules=True
    )
    load_reaction_rules.cache_clear()

    assert useful_reactor._automorphism_filter is False
    assert later_reactor._automorphism_filter is not False


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
    (reactor,) = load_reaction_rules(str(rules_path), detect_symmetric_rules=True)
    load_reaction_rules.cache_clear()

    assert reactor._automorphism_filter is False


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
    assert is_useful_symmetric_reaction_rule(rule_smarts)


@pytest.mark.parametrize(
    ("metal", "metal_handle"),
    [
        pytest.param("Zn", "[Zn:3]-[Br:5]", id="zinc"),
        pytest.param("Cu", "[Cu:3]", id="copper"),
        pytest.param("Li", "[Li:3]", id="lithium"),
        pytest.param("Al", "[Al:3](-[Cl:5])-[Cl:6]", id="aluminium"),
    ],
)
def test_extra_organometallic_metals_need_complementary_precursor_handle(
    metal, metal_handle
):
    rule_smarts = f"[C:1]-[C:2]>>[C:1]-{metal_handle}.[C:2]-[N:4]"

    assert metal in rule_smarts
    assert not is_useful_symmetric_reaction_rule(rule_smarts)


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
            "[C:1]-[C:2]=[C:5]-[C:6]>>[C:4]=[C:5]-[C:6].[C:1]-[C:2]=[C:3]",
            id="cross-metathesis",
        ),
        pytest.param(
            "[C:1]-[C:2]>>[C:1]=[O:3].[C:2]-[Cl:4]",
            id="benzaldehyde-benzyl-chloride",
        ),
        pytest.param(
            "[c:1]-[C:2]=[C:3]-[c:4]>>"
            "[Br:5]-[c:1].[C:2](-[Br:6])=[C:3](-[Br:7]).[Br:8]-[c:4]",
            id="dibromoalkene-halobenzene",
        ),
        pytest.param(
            "[c:1]-[N:2]=[N:3]-[c:4]>>[c:1]-[N+:5]#[N:6].[c:4]-[H:7]",
            id="diazotisation-coupling",
        ),
        pytest.param(
            "[n:1]:1:[c:2](:[c:3]):[c:5](:[c:4]):[n:6]:[c:8]:1-[C:7]>>"
            "[N:1]-[c:2](:[c:3]):[c:5](:[c:4])-[N:6]."
            "[C:7]-[C:8]#[N:9]",
            id="nitrile-decyanation-like",
        ),
    ],
)
def test_useful_symmetric_rule_detects_expanded_precursor_families(rule_smarts):
    assert is_useful_symmetric_reaction_rule(rule_smarts)


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
    assert not is_useful_symmetric_reaction_rule(rule_smarts)
