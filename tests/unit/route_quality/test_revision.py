"""Tests for deterministic protection/deprotection route revision."""

import csv
from pathlib import Path

import pytest
from chython import smiles

from synplan.routes.io import make_json
from synplan.routes.quality.protection.config import ProtectionRevisionConfig
from synplan.routes.quality.protection.revision import (
    ProtectionFragmentCatalog,
    ProtectionRouteReviser,
)
from synplan.routes.route_cgr import compose_route_cgr


def _esterification_with_competing_alcohol():
    return smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )


def _esterification_then_spectator_reaction():
    return {
        0: _esterification_with_competing_alcohol(),
        1: smiles(
            "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[Cl:9]>>"
            "[CH3:1][C:2](=[O:3])[O:5][CH:6]([Cl:9])[CH2:7][OH:8]"
        ),
    }


def _two_step_competing_carbonyl_route():
    return {
        0: smiles(
            "[O:20]=[C:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH2:7]1>>"
            "[OH:20][CH:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH2:7]1"
        ),
        1: smiles(
            "[OH:20][CH:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH2:7]1>>"
            "[CH:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH:7]=1.[OH2:20]"
        ),
    }


def _catalog():
    return ProtectionFragmentCatalog.from_config(ProtectionRevisionConfig())


def _protection_rows():
    with open(
        ProtectionRevisionConfig().protection_group_templates_path,
        encoding="utf-8",
        newline="",
    ) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


PROTECTION_SCHEMA = [
    "rule_name",
    "reaction_class",
    "smarts",
    "atoms_to_keep",
    "atoms_to_add",
    "protected_example",
    "cleaved_example",
    "decoys",
    "h2o",
    "bases",
    "nucleophiles",
    "electrophiles",
    "reduction",
    "oxidation",
    "p_mol",
]


def _catalog_from_templates(path: Path):
    return ProtectionFragmentCatalog(
        str(path),
        ProtectionRevisionConfig().reactive_function_label_mapping_path,
    )


def test_unified_csv_schema_and_rule_names():
    rows = _protection_rows()
    assert rows
    assert list(rows[0]) == PROTECTION_SCHEMA
    with open(
        ProtectionRevisionConfig().protection_group_templates_path,
        encoding="utf-8",
    ) as fh:
        assert fh.readline().startswith("rule_name\treaction_class\tsmarts")

    assert [row["rule_name"] for row in rows] == sorted(
        row["rule_name"] for row in rows
    )
    assert len({row["rule_name"] for row in rows}) == len(rows)
    assert all(row["rule_name"] for row in rows)
    assert all(row["reaction_class"] for row in rows)
    assert "substructure" not in rows[0]
    assert "source" not in rows[0]
    assert "license" not in rows[0]
    assert "source_url" not in rows[0]
    assert "template" not in rows[0]
    assert "label" not in rows[0]
    assert "deprotection_class" not in rows[0]
    assert "example_reagent" not in rows[0]


def test_fragment_catalog_accepts_valid_empty_tsv_schema(tmp_path):
    path = tmp_path / "protection_group_templates.csv"
    path.write_text("\t".join(PROTECTION_SCHEMA) + "\n")

    catalog = _catalog_from_templates(path)

    assert catalog.fragments == ()


def test_fragment_catalog_rejects_comma_delimited_legacy_schema(tmp_path):
    path = tmp_path / "protection_group_templates.csv"
    path.write_text(",".join(PROTECTION_SCHEMA) + "\n")

    with pytest.raises(ValueError, match="tab-delimited"):
        _catalog_from_templates(path)


def test_fragment_catalog_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "protection_group_templates.csv"
    path.write_text("\t".join(PROTECTION_SCHEMA[:-1]) + "\n")

    with pytest.raises(ValueError, match="missing columns: p_mol"):
        _catalog_from_templates(path)


def test_fragment_catalog_rejects_forbidden_legacy_columns(tmp_path):
    path = tmp_path / "protection_group_templates.csv"
    path.write_text("\t".join([*PROTECTION_SCHEMA, "template"]) + "\n")

    with pytest.raises(ValueError, match="unexpected columns: template"):
        _catalog_from_templates(path)


def test_unified_csv_condition_metadata_and_p_mol():
    rows = _protection_rows()
    by_rule = {row["rule_name"]: row for row in rows}

    assert by_rule["amine_acyl"]["h2o"].startswith("{'pH < 1")
    assert by_rule["amine_benzyl"]["reduction"].startswith("{'H2 / Ni': 2")
    assert by_rule["amine_phth"]["h2o"].startswith("{'pH < 1")
    assert by_rule["amine_tfa"]["nucleophiles"].startswith("{'RLi': 2")
    assert by_rule["amine_tosyl"]["reduction"].startswith("{'H2 / Ni': 0")
    assert by_rule["amine_tritil"]["electrophiles"].startswith("{'RCOCl': 0")
    assert by_rule["amine_boc"]["p_mol"] == (
        "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"
    )
    assert by_rule["carbonyl_dimethoxy"]["p_mol"] == "CO.CO"
    assert by_rule["carbonyl_dimethoxy"]["h2o"].startswith("{'pH < 1")
    assert by_rule["carbonyl_dioxane"]["oxidation"].startswith("{'KMnO4': 0")
    assert by_rule["carbonyl_dioxolane"]["oxidation"].startswith("{'KMnO4': 0")
    assert by_rule["carbonyl_dithiane"]["electrophiles"].startswith("{'RCOCl': 0")
    assert by_rule["carbonyl_dithiolane"]["reduction"].startswith("{'H2 / Ni': 2")
    assert by_rule["carbonyl_nn_dimethylhydrazone"]["bases"].startswith(
        "{'LDA': 2"
    )
    assert by_rule["carbonyl_nn_dimethylhydrazone"]["p_mol"] == "CN(C)N"
    assert by_rule["carboxyl__oxazoline"]["electrophiles"].startswith(
        "{'RCOCl': 2"
    )
    assert by_rule["carboxyl_benzyl"]["reduction"].startswith("{'H2 / Ni': 2")
    assert by_rule["carboxyl_boc"]["oxidation"].startswith("{'KMnO4': 0")
    assert by_rule["carboxyl_methoxy"]["nucleophiles"].startswith("{'RLi': 2")
    assert by_rule["carboxyl_s_boc"]["h2o"].startswith("{'pH < 1")
    assert by_rule["hydroxyl_mom"]["h2o"].startswith("{'pH < 1")
    assert by_rule["hydroxyl_benzyl"]["h2o"].startswith("{'pH < 1")
    assert by_rule["hydroxyl_tbs"]["h2o"].startswith("{'pH < 1")
    assert by_rule["hydroxyl_acyl"]["nucleophiles"].startswith("{'RLi': 2")
    assert by_rule["hydroxyl_piv"]["bases"].startswith("{'LDA': 0")
    assert by_rule["hydroxyl_benzoate"]["h2o"].startswith("{'pH < 1")
    assert by_rule["diol_12_acetone"]["h2o"].startswith("{'pH < 1")
    assert by_rule["diol_13_acetone"]["h2o"].startswith("{'pH < 1")
    assert by_rule["diol_12_benzylidene"]["h2o"].startswith("{'pH < 1")
    assert by_rule["diol_13_benzylidene"]["h2o"].startswith("{'pH < 1")
    assert by_rule["hydroxyl_benzyl"]["p_mol"] == "BrCc1ccccc1"
    for row in rows:
        assert smiles(row["p_mol"]) is not None

    benzylidene = by_rule["amine_benzylidene"]
    assert benzylidene["reaction_class"] == "amine"
    assert benzylidene["smarts"] == (
        "[N;D2:1]=;!@[C;D2;z2;x1]-[C;a;r6]:2:[C;D2]:[C;D2]:"
        "[C;D2]:[C;D2]:[C;D2]:2"
    )
    assert benzylidene["p_mol"] == "O=Cc1ccccc1"


def test_stability_legend_markdown_documents_source_and_colors():
    source = Path(
        "synplan/routes/quality/protection/data/protecting_group_stability.md"
    ).read_text()

    assert "green" in source
    assert "yellow" in source
    assert "orange" in source
    assert "https://www.organic-chemistry.org/protectivegroups/" in source
    assert "T. W. Green" in source
    assert "Protective Groups in Organic Synthesis" in source


def test_unified_csv_deduplicates_by_rule_name():
    rows = _protection_rows()
    dithiane_rows = [row for row in rows if row["rule_name"] == "carbonyl_dithiane"]

    assert len(dithiane_rows) == 1
    assert dithiane_rows[0]["reaction_class"] == "carbonyl"


def test_fragment_catalog_loads_unified_csv_and_marks_data_only_families():
    catalog = _catalog()
    loaded_rules = {fragment.rule_name for fragment in catalog.fragments}

    alcohol_candidates = catalog.candidates_for_fg("PrimaryAlcoholAliphatic")
    aldehyde_candidates = catalog.candidates_for_fg("Aldehyde_SaturatedAliphatic")
    acid_candidates = catalog.candidates_for_fg("Acid_SaturatedAliphatic")
    amino_acid_candidates = catalog.candidates_for_fg(
        "NonProlineAlphaAminoAcid_unprotected"
    )

    assert alcohol_candidates
    assert aldehyde_candidates
    assert not acid_candidates
    assert not catalog.candidates_for_fg("ThiolAliphatic")
    assert not catalog.candidates_for_fg("DiolAliphatic")
    assert amino_acid_candidates
    assert "carboxyl_benzyl" in loaded_rules
    assert "thiol_benzyl" in loaded_rules
    assert "diol_12_acetone" in loaded_rules
    assert any(
        diagnostic.reason == "unsupported_protection_family"
        and diagnostic.rule_name == "carboxyl_benzyl"
        for diagnostic in catalog.diagnostics
    )
    assert any(
        candidate.rule_name == "amine_boc"
        for candidate in amino_acid_candidates
    )
    assert any(
        candidate.strategy == "carbonyl_acetal"
        for candidate in aldehyde_candidates
    )
    assert any(
        candidate.rule_name == "amine_benzylidene"
        and candidate.strategy == "single_anchor"
        and candidate.p_mol == "O=Cc1ccccc1"
        for candidate in catalog.candidates_for_fg("SecondaryAmineAliphatic")
    )
    assert any(
        candidate.rule_name == "amine_boc"
        for candidate in catalog.candidates_for_fg("Benzylamine_primary")
    )
    assert any(
        candidate.rule_name == "amine_boc"
        for candidate in catalog.candidates_for_fg("HeteroBenzylamine_primary")
    )
    assert all(
        candidate.rule_name != "carbonyl_nn_dimethylhydrazone"
        for candidate in aldehyde_candidates
    )
    assert all(
        candidate.rule_name
        not in {
            "carboxyl__oxazoline",
            "carboxyl_benzyl",
            "carboxyl_boc",
            "carboxyl_methoxy",
            "carboxyl_s_boc",
        }
        for candidate in acid_candidates
    )
    assert all(
        candidate.rule_name != "carboxyl_trioxabicyclooctane"
        for candidate in acid_candidates
    )


def test_imine_rules_are_supported_single_anchor_with_double_attachment_bonds():
    catalog = _catalog()
    candidates = {
        candidate.rule_name: candidate
        for candidate in catalog.candidates_for_fg("SecondaryAmineAliphatic")
    }

    for rule_name in (
        "amine_benzhydrylidene",
        "amine_benzylidene",
        "amine_dde_imine",
        "amine_ivdde_imine",
    ):
        fragment = candidates[rule_name]
        assert fragment.strategy == "single_anchor"
        assert fragment.attachment_bond_order == 2
        assert fragment.attachment_bonds == ((fragment.attachment_atom, 2),)


def test_single_anchor_attachment_preserves_imine_double_bond():
    catalog = _catalog()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )
    fragment = next(
        candidate
        for candidate in catalog.candidates_for_fg("SecondaryAmineAliphatic")
        if candidate.rule_name == "amine_benzylidene"
    )
    atom_mapping = {
        atom_id: idx + 100
        for idx, atom_id in enumerate(sorted(fragment.molecule._atoms))
    }

    protected = reviser._attach_fragment(
        smiles("[NH2:1][CH3:2]"),
        1,
        fragment,
        atom_mapping,
    )

    assert protected is not None
    attached_atom = atom_mapping[fragment.attachment_atom]
    assert protected._bonds[1][attached_atom].order == 2


def test_single_anchor_attachment_keeps_single_bond_for_alkyl_protections():
    catalog = _catalog()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )
    fragment = next(
        candidate
        for candidate in catalog.candidates_for_fg("SecondaryAmineAliphatic")
        if candidate.rule_name == "amine_benzyl"
    )
    atom_mapping = {
        atom_id: idx + 100
        for idx, atom_id in enumerate(sorted(fragment.molecule._atoms))
    }

    protected = reviser._attach_fragment(
        smiles("[NH2:1][CH3:2]"),
        1,
        fragment,
        atom_mapping,
    )

    assert protected is not None
    attached_atom = atom_mapping[fragment.attachment_atom]
    assert protected._bonds[1][attached_atom].order == 1


def test_charged_fragment_attachment_preserves_nosyl_nitro_charge():
    catalog = _catalog()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )
    fragment = next(
        candidate
        for candidate in catalog.candidates_for_fg(
            "Amine_Primary_SaturatedAliphatic"
        )
        if candidate.rule_name == "amine_nosyl"
    )
    atom_mapping = {
        atom_id: idx + 100
        for idx, atom_id in enumerate(sorted(fragment.molecule._atoms))
    }

    protected = reviser._attach_fragment(
        smiles("[NH2:1][CH3:2]"),
        1,
        fragment,
        atom_mapping,
    )

    assert protected is not None
    assert any(
        atom.atomic_symbol == "N" and getattr(atom, "charge", 0) == 1
        for _, atom in protected.atoms()
    )
    assert any(
        atom.atomic_symbol == "O" and getattr(atom, "charge", 0) == -1
        for _, atom in protected.atoms()
    )


def test_reviser_inserts_protection_deprotection_and_improves_score():
    route = {0: _esterification_with_competing_alcohol()}
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )

    result = reviser.revise_route(route, route_id=0)

    assert result.accepted
    assert result.revised_score > result.original_score
    assert len(result.route) == 3
    action = result.actions[0]
    assert action.fg_name == "PrimaryAlcoholAliphatic"
    assert action.fg_atoms == (8,)
    assert action.anchor_atom == 8
    assert action.new_step_ids == (0, 1, 2)
    assert action.p_mol != "None"
    assert action.p_mol.endswith("Cl")
    assert len(result.route[0].reactants) == 2
    assert result.route[0].reactants[1].meta["added_protecting_group"] is True
    assert set(result.route_metadata) == {0, 1, 2}
    assert result.route_metadata[0]["protection_revision"]["p_mol"] == action.p_mol
    assert "conditions" in result.route_metadata[0]["protection_revision"]
    assert "label" not in result.route_metadata[0]["protection_revision"]
    assert "deprotection_class" not in result.route_metadata[0]["protection_revision"]
    assert "example_reagent" not in result.route_metadata[0]["protection_revision"]
    assert "template" not in result.route_metadata[0]["protection_revision"]
    assert compose_route_cgr({0: result.route}, 0) is not None


def test_reviser_delays_deprotection_while_site_is_downstream_spectator():
    route = _esterification_then_spectator_reaction()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )

    result = reviser.revise_route(route, route_id=0)

    assert result.accepted
    assert result.revised_score > result.original_score
    assert len(result.route) == 4
    action = result.actions[0]
    assert action.new_step_ids == (0, 1, 3)
    assert result.route_metadata[0]["protection_revision"]["role"] == "protection"
    assert (
        result.route_metadata[1]["protection_revision"]["role"]
        == "protected_transformation"
    )
    assert (
        result.route_metadata[2]["protection_revision"]["role"]
        == "protected_downstream_transformation"
    )
    assert result.route_metadata[3]["protection_revision"]["role"] == "deprotection"

    original_max_atom = max(
        atom_id
        for reaction in route.values()
        for molecule in list(reaction.reactants) + list(reaction.products)
        for atom_id in molecule._atoms
    )
    assert max(result.route[2].products[0]._atoms) > original_max_atom
    assert max(result.route[3].products[0]._atoms) <= original_max_atom
    assert compose_route_cgr({0: result.route}, 0) is not None


def test_reviser_protects_mapped_competing_carbonyl_route():
    route = _two_step_competing_carbonyl_route()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )

    scan = reviser.scanner.scan_route(route, detailed=True)
    step_0_ketone = next(
        interaction
        for interaction in scan.interactions
        if interaction.step_id == 0
        and interaction.fg_name == "KetoneAliphaticCyclic"
    )
    step_1_ketone = next(
        interaction
        for interaction in scan.interactions
        if interaction.step_id == 1
        and interaction.fg_name == "KetoneAliphaticCyclic"
    )
    original_score, _ = reviser.scorer.score_route(route)

    assert step_0_ketone.fg_atoms == (4,)
    assert step_0_ketone.anchor_atom == 4
    assert step_0_ketone.reacting_fg == "KetoneAliphaticCyclic"
    assert step_0_ketone.severity == "competing"
    assert step_1_ketone.severity == "compatible"
    assert original_score == 0.75

    result = reviser.revise_route(route, route_id=0)

    assert result.accepted
    assert result.revised_score > result.original_score
    action = result.actions[0]
    assert action.fg_name == "KetoneAliphaticCyclic"
    assert action.rule_name == "carbonyl_dioxolane"
    assert action.protection_class == "carbonyl"
    assert compose_route_cgr({0: result.route}, 0) is not None


def test_carbonyl_fragment_edit_validates():
    catalog = _catalog()
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )

    carbonyl_fragment = next(
        candidate
        for candidate in catalog.candidates_for_fg("Aldehyde_SaturatedAliphatic")
        if candidate.rule_name == "carbonyl_dioxolane"
    )
    carbonyl_route = {
        0: smiles("[CH3:1][CH:2]=[O:3]>>[CH3:1][CH:2]=[O:3]")
    }
    carbonyl_mapping = reviser._fresh_fragment_atom_mapping(
        carbonyl_route,
        carbonyl_fragment,
    )
    protected_carbonyl = reviser._attach_fragment(
        smiles("[CH3:1][CH:2]=[O:3]"),
        2,
        carbonyl_fragment,
        carbonyl_mapping,
    )

    assert protected_carbonyl is not None
    assert 2 in protected_carbonyl._atoms
    assert 3 not in protected_carbonyl._atoms


def test_reviser_metadata_is_exportable_to_route_json():
    route = {0: _esterification_with_competing_alcohol()}
    reviser = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )
    result = reviser.revise_route(route, route_id=0)

    routes_json = make_json(
        {0: result.route},
        route_metadata={0: result.route_metadata},
    )

    def reaction_nodes(node):
        if not node:
            return []
        found = [node] if node.get("type") == "reaction" else []
        for child in node.get("children", []):
            found.extend(reaction_nodes(child))
        return found

    def mol_nodes(node):
        if not node:
            return []
        found = [node] if node.get("type") == "mol" else []
        for child in node.get("children", []):
            found.extend(mol_nodes(child))
        return found

    nodes = reaction_nodes(routes_json[0])
    assert any("protection_revision" in node for node in nodes)
    assert any(node.get("added_protecting_group") for node in mol_nodes(routes_json[0]))
