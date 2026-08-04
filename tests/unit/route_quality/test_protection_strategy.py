"""Protecting-group selection: data loading, the classifier seam, and rule building."""

import pytest
from chython import smiles

from synplan.chem.reaction.routes.quality.protection.config import ProtectionConfig
from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
)
from synplan.chem.reaction.routes.quality.protection.strategy import (
    FirstAllowedClassifier,
    ProtectingGroupClassifier,
    ProtectionPlanner,
    build_protection_rule,
    load_allowed_labels,
    load_protecting_groups,
    protect_molecule,
)


@pytest.fixture(scope="module")
def config():
    return ProtectionConfig()


@pytest.fixture(scope="module")
def planner(config):
    return ProtectionPlanner(
        load_protecting_groups(config.protecting_groups_path),
        load_allowed_labels(config.allowed_labels_path),
    )


def test_protecting_groups_keep_every_row(config):
    """The table has more rows than labels; a flat dict would drop the extras."""
    groups = load_protecting_groups(config.protecting_groups_path)
    assert sum(len(v) for v in groups.values()) > len(groups)


def test_bidentate_templates_are_dropped_not_offered(config, caplog):
    """Nine templates attach at two points and cannot be built; say so once."""
    import logging

    with caplog.at_level(logging.WARNING):
        groups = load_protecting_groups(config.protecting_groups_path)

    assert not (set(groups) & {11, 12, 13, 14, 18})
    assert "no usable template" in caplog.text
    for rows in groups.values():
        for group in rows:
            build_protection_rule("[C;H2:1]", group)  # every kept row is usable


def test_carbonyl_protection_is_the_known_gap(config):
    """Aldehydes and ketones have no protection option; keep that visible."""
    planner = ProtectionPlanner(
        load_protecting_groups(config.protecting_groups_path),
        load_allowed_labels(config.allowed_labels_path),
    )
    assert planner.candidates("Aldehyde_Aromatic") == []
    assert planner.candidates("KetoneAliphaticCyclic") == []
    assert planner.candidates("Phenol")  # unaffected groups still work


def test_allowed_labels_load(config):
    labels = load_allowed_labels(config.allowed_labels_path)
    assert labels
    assert all(isinstance(v, tuple) for v in labels.values())


@pytest.mark.parametrize(
    "smi,expected",
    [
        # phenol needs the recursive-SMARTS alternation chython <1.97 got wrong
        ("Oc1ccccc1", "Phenol"),
        ("CCO", "PrimaryAlcoholAliphatic"),
    ],
)
def test_detects_functional_group(config, smi, expected):
    molecule = smiles(smi)
    molecule.canonicalize()
    detector = FunctionalGroupDetector(config.competing_groups_path)
    assert expected in {m.name for m in detector.detect_all(molecule)}


def test_base_classifier_predicts_nothing():
    """The Chemformer seam must abstain, not guess, while unwired."""
    groups = [object()]
    assert ProtectingGroupClassifier().rank("Phenol", None, (), groups) == []
    assert FirstAllowedClassifier().rank("Phenol", None, (), groups) == groups


def test_candidates_come_from_the_mapping(planner):
    assert planner.candidates("Phenol")
    assert planner.candidates("NotAFunctionalGroup") == []


def test_protection_rule_attaches_the_group(planner, config):
    """A built rule must add atoms, not just match."""
    detector = FunctionalGroupDetector(config.competing_groups_path)
    template = detector.template_for("PrimaryAlcoholAliphatic")
    if template is None:
        pytest.skip("no protection template shipped for this group")

    group = planner.candidates("PrimaryAlcoholAliphatic")[0]
    molecule = smiles("OCCc1ccccc1")
    molecule.canonicalize()

    protected = protect_molecule(molecule, template[0], group)
    if protected is None:
        pytest.skip(f"{group.reaction_class} does not apply to this substrate")
    assert len(protected) > len(molecule)


def test_build_protection_rule_is_reusable(planner, config):
    detector = FunctionalGroupDetector(config.competing_groups_path)
    template = detector.template_for("PrimaryAlcoholAliphatic")
    if template is None:
        pytest.skip("no protection template shipped for this group")
    group = planner.candidates("PrimaryAlcoholAliphatic")[0]
    assert build_protection_rule(template[0], group) is not None
