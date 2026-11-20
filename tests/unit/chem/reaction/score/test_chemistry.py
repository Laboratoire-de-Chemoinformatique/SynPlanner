"""Applicability, balancing and exhaustive stereo contracts without models."""

from copy import deepcopy

import pytest
from chython import smiles
from rdkit import Chem

from synplan.chem.reaction.score import AIMNetConfig
from synplan.chem.reaction.score.aimnet import preparation as chemistry
from synplan.chem.reaction.score.aimnet.config import MolecularCalculationError
from synplan.chem.reaction.score.aimnet.preparation import (
    conserved,
    prepare_reaction,
    reaction_variants,
    stereoisomers,
)


def test_balancer_adds_water_and_preserves_original():
    source = smiles("CC(=O)O.CCO>>CC(=O)OCC")
    before = deepcopy(source)
    info, left, right = prepare_reaction(source, 0.8)
    assert source == before and source.meta == before.meta
    assert conserved(left, right)
    assert info["balancing_confidence"] == 0.94
    assert info["inferred_species"]["products"] == [{"smiles": "O", "count": 1}]
    assert info["original_reaction"] != info["balanced_reaction"]


def test_balancing_confidence_boundary():
    reaction = smiles("CC(=O)O.CCO>>CC(=O)OCC")
    prepare_reaction(reaction, 0.94)
    with pytest.raises(MolecularCalculationError, match="confidence"):
        prepare_reaction(reaction, 0.941)


def test_charge_is_counted_even_when_negative():
    assert not conserved(["[OH-]", "[H+]"], ["[OH-]"])
    assert not conserved(["[NH4+]"], ["N"])
    assert conserved(["[OH-]", "[H+]"], ["O"])
    assert not conserved(["[2H]O[2H]"], ["O"])


def test_added_species_are_rechecked(monkeypatch):
    monkeypatch.setattr(
        chemistry, "rebalance_reaction", lambda *a, **kw: smiles("[Na+].CCO>>CCO.[Na+]")
    )
    with pytest.raises(MolecularCalculationError, match="Na"):
        prepare_reaction(smiles("CC(=O)O.CCO>>CC(=O)OCC"), 0.8)


def test_reagents_do_not_filter_or_contribute():
    info, left, right = prepare_reaction(smiles("CCO>[Pd]>CCO"), 0.8)
    assert left == right == ["CCO"]
    assert "Pd" in info["original_reaction"]


def test_already_balanced_skips_balancer(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Already-balanced input must not infer reagents")

    monkeypatch.setattr(chemistry, "rebalance_reaction", unexpected)
    info, left, right = prepare_reaction(smiles("CC(=O)O.CCO>>CC(=O)OCC.O"), 1.0)
    assert conserved(left, right)
    assert info["balancing_confidence"] == 1.0
    assert info["inferred_species"] == {"reactants": [], "products": []}


def test_fixed_stereo_and_ez_enumeration():
    assert len(list(stereoisomers("CC(O)C(=O)O"))) == 2
    assert list(stereoisomers("C[C@H](O)C(=O)O")) == ["C[C@H](O)C(=O)O"]
    assert len(list(stereoisomers("FC=CCl"))) == 2
    assert list(stereoisomers("F/C=C/Cl")) == ["F/C=C/Cl"]
    # Only the second center is enumerated; the first retains its configuration.
    products = list(stereoisomers("C[C@H](O)C(F)Cl"))
    assert len(products) == 2
    labels = []
    for text in products:
        mol = Chem.MolFromSmiles(text)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        labels.append(mol.GetAtomWithIdx(1).GetProp("_CIPCode"))
    assert len(set(labels)) == 1


def test_no_256_cap_and_no_reaction_cartesian_truncation():
    molecule = "CC(F)C(Cl)C(Br)C(I)C(O)C(N)C(S)C(C#N)C(C=O)CO"
    assert sum(1 for _ in stereoisomers(molecule)) == 512
    variants = reaction_variants([molecule], ["CCO"])
    assert iter(variants) is variants
    assert sum(1 for _ in variants) == 512


def test_symmetric_species_permutations_and_stoichiometry():
    # RR, RS, SS (RS and SR are the same two-species side).
    variants = list(reaction_variants(["CC(O)F", "CC(O)F"], ["O"]))
    assert len(variants) == 3
    assert all(len(left) == 2 for left, _ in variants)
    # Two equivalent centers have only three stereoisomers, including meso.
    assert len(list(stereoisomers("CC(Cl)C(Cl)C"))) == 3


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": 0},
        {"pressure": float("nan")},
        {"n_conformers": 0},
        {"max_steps": -1},
        {"symmetry_numbers": {"O": 0}},
        {"device": "tpu"},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        AIMNetConfig(**settings)


@pytest.mark.parametrize("charge", [-3, -2, 3, 4])
def test_loqi_charge_vocabulary_and_atomic_bypass(charge):
    from rdkit import Chem

    from synplan.chem.reaction.score.aimnet.species import validate_loqi_charges

    molecule = Chem.MolFromSmiles("C[Si-3]")
    molecule.GetAtomWithIdx(1).SetFormalCharge(charge)
    molecule.UpdatePropertyCache(strict=False)
    if -2 <= charge <= 3:
        validate_loqi_charges(molecule)
    else:
        with pytest.raises(MolecularCalculationError) as caught:
            validate_loqi_charges(molecule)
        assert caught.value.reason == "unsupported_charge"
        assert caught.value.category == "filtered_out"
    validate_loqi_charges(Chem.MolFromSmiles("[N-3]"))


def test_unsupported_charge_does_not_abort_collection():
    from synplan.chem.reaction.score import AIMNetReactionAssessor

    from .test_aimnet import Calculator

    calculator = Calculator()
    results = AIMNetReactionAssessor(
        cache_path=None, calculator=calculator
    ).assess_reactions([smiles("C[Si-3]>>C[Si-3]"), smiles("CCO>>CCO")])
    assert results[0]["failure"]["reason"] == "unsupported_charge"
    assert results[0]["status"] == "skipped"
    assert results[1]["status"] == "ok"
    assert calculator.calls == ["CCO"]


def test_direct_backend_rejects_charge_before_loading_models(monkeypatch):
    from synplan.chem.reaction.score import AIMNetConfig
    from synplan.chem.reaction.score.aimnet.molecular import AIMNetMolecularCalculator

    backend = AIMNetMolecularCalculator(AIMNetConfig())
    monkeypatch.setattr(
        backend, "identity", lambda: pytest.fail("models must not load")
    )
    with pytest.raises(MolecularCalculationError, match="LoQI supports"):
        backend.calculate("C[Si-3]")


def test_balanced_completion_rechecks_loqi_charges(monkeypatch):
    monkeypatch.setattr(
        chemistry, "rebalance_reaction", lambda *a, **kw: smiles("C[Si-3]>>C[Si-3]")
    )
    with pytest.raises(MolecularCalculationError) as caught:
        prepare_reaction(smiles("CC(=O)O.CCO>>CC(=O)OCC"), 0.8)
    assert caught.value.reason == "unsupported_charge"
    assert caught.value.reaction_context["balanced_reaction"]
