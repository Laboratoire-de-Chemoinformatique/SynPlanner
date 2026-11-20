"""Independent standard states, reference cycles, phase availability and CSV resumes."""

from math import log
from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet.config import MolecularCalculationError
from synplan.chem.reaction.score.aimnet.molecular import EV_TO_KCAL_MOL
from synplan.chem.reaction.score.aimnet.phase import R_KCAL, separates_ions
from synplan.chem.reaction.score.aimnet.references import ReferenceTable

from .test_aimnet import Calculator, route


def water_config(**kwargs):
    return AIMNetConfig(
        phase="cpcm_water",
        cpcm_checkpoint=Path("water.pt"),
        cpcm_sha256="1" * 64,
        concentration_mol_l=1,
        **kwargs,
    )


def test_reference_cycle_and_reference_digest():
    config = water_config()
    table, digest = ReferenceTable.load("bundled", config)

    def gas(s):
        return {"gibbs_eV": 1.0, "enthalpy_eV": 2.0, "energy_eV": 0.75}

    def water(s):
        return {"gibbs_eV": 0.5, "enthalpy_eV": 0.8, "energy_eV": 0.3}

    row = table.molecular_values("[F-]", "cpcm_water", gas, water)
    corrected_gas = table.molecular_values("[F-]", "gas", gas, water)
    assert (
        row["gibbs_eV"] - corrected_gas["gibbs_eV"]
    ) * EV_TO_KCAL_MOL == pytest.approx(-104.4)
    assert row["energy_eV"] == 0.3
    shift = row["corrections"][0]
    assert row["enthalpy_eV"] - row["gibbs_eV"] == pytest.approx(
        0.3 + (shift["delta_h_kcal_mol"] - shift["delta_g_kcal_mol"]) / EV_TO_KCAL_MOL
    )
    assert row["corrections"][1]["entropy_convention"] == "retain CPCM species entropy"
    assert len(digest) == 64
    with pytest.raises(ValueError, match="temperature"):
        ReferenceTable.load("bundled", config.model_copy(update={"temperature": 310}))


def test_derived_carbonate_enters_only_the_aqueous_phase():
    config = water_config(also_calculate_gas=True, reference_table="bundled")
    service = AIMNetReactionAssessor(
        config, None, calculator=Calculator(), gas_calculator=Calculator()
    )
    r = service.assess([route("O=C([O-])[O-].[H+]>>O=C([O-])O")])[0]
    row = r.steps[0].thermochemistry
    assert row["additional_phases"]["gas"]["failure"]["reason"] == "gas_unbound"
    result = row["variants"][0]
    assert result["delta_e"] is None
    assert result["delta_g"] == pytest.approx(-R_KCAL * 298.15 * log(10) * 10.33)
    assert result["delta_h"] == pytest.approx(-14.85 / 4.184)
    assert result["species_corrections"]


def test_independent_phase_failures():
    error = MolecularCalculationError("optimization", "not_converged", "test")
    row = AIMNetReactionAssessor(
        water_config(also_calculate_gas=True),
        None,
        calculator=Calculator(fail=error),
        gas_calculator=Calculator(),
    ).assess_reactions([smiles("CCO>>CCO")])[0]
    assert row["status"] == "failed"
    assert row["additional_phases"]["gas"]["status"] == "ok"
    assert separates_ions({"[Cl-]": 1, "[H+]": 1, "Cl": -1})
    assert not separates_ions({"[Cl-]": 1, "[Br-]": -1})


def test_phase_difference_and_reversible_standard_state():
    a = AIMNetReactionAssessor(
        water_config(also_calculate_gas=True, solvent_concentration_mol_l=55.5084),
        None,
        calculator=Calculator(),
        gas_calculator=Calculator(),
    )
    row = a.assess_reactions([smiles("CC(=O)O.CCO>>CCOC(C)=O.O")])[0]
    v = row["variants"][0]
    shift = R_KCAL * 298.15 * log(55.5084)
    assert v["delta_delta_g_solv_kcal_mol"] == pytest.approx(shift)
    assert v["raw_delta_g"] + shift == pytest.approx(v["delta_g"])
    assert v["corrections"][0]["delta_h_kcal_mol"] == 0


def test_specified_stereo_and_mirror_policy():
    from synplan.chem.reaction.score.aimnet.geometry import matches_stereo
    from synplan.chem.reaction.score.aimnet.species import parse_species

    observed = parse_species("F[C@H](Cl)Br")
    assert matches_stereo(observed, parse_species("FC(Cl)Br"), True, False)
    assert not matches_stereo(observed, parse_species("F[C@@H](Cl)Br"), True, False)
    assert matches_stereo(observed, parse_species("F[C@@H](Cl)Br"), True, True)
    assert not matches_stereo(observed, parse_species("F[13C@H](Cl)Br"), True, True)


def test_shared_generator_reuses_seed_pool_and_keeps_phase_models_separate(monkeypatch):
    from synplan.chem.reaction.score.aimnet.conformers import LoQIGenerator

    calls = []

    def sample(self, species, seed):
        calls.append((species, seed))
        return object()

    monkeypatch.setattr(LoQIGenerator, "_sample", sample)
    a = AIMNetReactionAssessor(water_config(also_calculate_gas=True), None)
    assert a.calculator is not a.gas.calculator
    assert a.calculator._generate("CCO") is a.gas.calculator._generate("CCO")
    assert len(calls) == 1


def test_quality_records_convention_cancellation_and_unanchored_ion():
    from synplan.chem.reaction.score.aimnet.phase import phase_quality

    net = {"[NH4+]": 1, "[Cl-]": 1, "CC(Cl)F": 1, "C[O-]": -1}
    records = {s: {} for s in net}
    records["[NH4+]"] = {"corrections": [{"reason": "base_pka"}]}
    records["[Cl-]"] = {"corrections": [{"reason": "ion_solvation"}]}
    result = phase_quality(net, records, "cpcm_water", "conformer")
    assert result["proton_convention_exposure"] == 0
    assert result["species_warnings"] == {
        "CC(Cl)F": ["unassigned_stereochemistry"],
        "C[O-]": ["unanchored_ion"],
    }


def test_derived_base_fallback_retains_failure_and_neutral_entropy():
    table, _ = ReferenceTable.load("bundled", water_config())

    def water(s):
        if s == "[NH4+]":
            raise MolecularCalculationError("conformers", "no_valid_minimum", "test")
        return {"energy_eV": 2.1, "enthalpy_eV": 2.3, "gibbs_eV": 2.2}

    row = table.molecular_values("[NH4+]", "cpcm_water", water, water)
    assert row["energy_eV"] is None
    assert row["reference_base_species"] == "N"
    assert row["entropy_eV_per_K"] == pytest.approx(0.1 / 298.15)
    assert row["corrections"][0]["raw_species_failure"]["reason"] == "no_valid_minimum"


def test_whole_mirror_for_azide_and_unresolved_alkene():
    from synplan.chem.reaction.score.aimnet.phase import enantiomer_identity
    from synplan.chem.reaction.score.aimnet.species import (
        canonical_species,
        parse_species,
    )

    def pair(a, b):
        return {
            canonical_species(parse_species(a)): -2,
            canonical_species(parse_species(b)): 2,
        }

    assert enantiomer_identity(pair("C[C@H](F)N=[N+]=[N-]", "C[C@@H](F)N=[N+]=[N-]"))
    assert not enantiomer_identity(pair("C[C@H](F)C=CC", "C[C@@H](F)C=CC"))
