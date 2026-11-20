"""Independent standard states, reference cycles, phase availability and CSV resumes."""

from math import log
from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet.config import MolecularCalculationError
from synplan.chem.reaction.score.aimnet.phase import R_KCAL, separates_ions

from .test_aimnet import Calculator


def water_config(**kwargs):
    return AIMNetConfig(
        phase="cpcm_water",
        cpcm_checkpoint=Path("water.pt"),
        cpcm_sha256="1" * 64,
        concentration_mol_l=1,
        **kwargs,
    )


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
