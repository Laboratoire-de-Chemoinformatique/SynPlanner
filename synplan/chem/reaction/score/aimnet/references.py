"""Experimental reference cycles applied to raw molecular records after calculation."""

from hashlib import sha256
from math import log
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .config import MolecularCalculationError
from .molecular import EV_TO_KCAL_MOL
from .phase import R_KCAL
from .species import canonical_species, parse_species


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1)
    value: FiniteFloat


class Equilibrium(Measurement):
    parent: str


class AcidIonization(Equilibrium):
    """Aqueous acid -> conjugate base + proton, with optional measured enthalpy."""

    enthalpy_kcal_mol: FiniteFloat | None = None


class FormationCycle(Measurement):
    """Measured reaction Gibbs energy (value) and enthalpy at the table standard state."""

    enthalpy_kcal_mol: FiniteFloat


class AcidDissociation(FormationCycle):
    parent: str
    gas_reference_pressure_Pa: FiniteFloat | None = Field(default=None, gt=0)


def _correction(reason, rule, delta_g, delta_h, entropy_convention):
    return {
        "reason": reason,
        "source": rule.source,
        "delta_g_kcal_mol": delta_g,
        "delta_h_kcal_mol": delta_h,
        "entropy_convention": entropy_convention,
    }


class ReferenceTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = Field(min_length=1)
    solvent: Literal["water"]
    temperature_K: FiniteFloat = Field(gt=0)
    concentration_mol_l: FiniteFloat = Field(gt=0)
    gas_unbound: tuple[str, ...] = ()
    ion_solvation: dict[str, Measurement]
    protonated_bases: dict[str, Equilibrium] = Field(default_factory=dict)
    deprotonated_acids: dict[str, AcidIonization] = Field(default_factory=dict)
    gas_acidities: dict[str, AcidDissociation] = Field(default_factory=dict)
    strong_acids: dict[str, FormationCycle] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_species(self):
        from .preparation import conserved

        for entries in (
            self.ion_solvation,
            self.protonated_bases,
            self.deprotonated_acids,
            self.gas_acidities,
            self.strong_acids,
        ):
            for species, entry in entries.items():
                if canonical_species(parse_species(species)) != species:
                    raise ValueError(
                        f"Reference key must be canonical isomeric SMILES: {species}"
                    )
                if isinstance(entry, (Equilibrium, AcidDissociation)):
                    parent = entry.parent
                    if canonical_species(parse_species(parent)) != parent:
                        raise ValueError(f"Invalid parent identity: {parent}")
                    acid, base = (
                        (species, parent)
                        if entries is self.protonated_bases
                        else (parent, species)
                    )
                    if not conserved([acid], [base, "[H+]"]):
                        raise ValueError(
                            f"Reference proton transfer does not conserve composition: {acid}, {base}"
                        )
        if any(s not in self.deprotonated_acids for s in self.gas_unbound):
            raise ValueError("An unbound species requires an aqueous parent reference")
        if "[H+]" not in self.ion_solvation:
            raise ValueError(
                "Reference table requires an explicit single-ion proton convention"
            )
        return self

    @classmethod
    def load(cls, setting, config):
        path = (
            Path(__file__).with_name("aqueous_references.json")
            if setting == "bundled"
            else Path(setting)
        )
        data = path.read_bytes()
        table = cls.model_validate_json(data)
        if (config.temperature, config.concentration_mol_l) != (
            table.temperature_K,
            table.concentration_mol_l,
        ):
            raise ValueError(
                "Reference table temperature and standard concentration must match the calculation"
            )
        return table, sha256(data).hexdigest()

    def _gas_acidity(self, species, rule, gas):
        bare, parent, proton = gas(species), gas(rule.parent), gas("[H+]")
        dg = (
            parent["gibbs_eV"] - proton["gibbs_eV"] - bare["gibbs_eV"]
        ) * EV_TO_KCAL_MOL + rule.value
        if rule.gas_reference_pressure_Pa is not None:
            pressure = (
                1000 * self.concentration_mol_l * 8.31446261815324 * self.temperature_K
            )
            dg += (
                R_KCAL
                * self.temperature_K
                * log(pressure / rule.gas_reference_pressure_Pa)
            )
        dh = (
            parent["enthalpy_eV"] - proton["enthalpy_eV"] - bare["enthalpy_eV"]
        ) * EV_TO_KCAL_MOL + rule.enthalpy_kcal_mol
        return _correction(
            "gas_acidity", rule, dg, dh, "use experimental enthalpy and Gibbs energy"
        )

    def _aqueous_values(self, species, gas, water):
        """Return a raw baseline, its species, and the aqueous correction to it."""
        if species == "[H+]":
            rule = self.ion_solvation[species]
            return (
                gas(species),
                species,
                _correction(
                    "proton_solvation",
                    rule,
                    rule.value,
                    rule.value,
                    "retain gas proton entropy",
                ),
            )

        if species in self.deprotonated_acids or species in self.protonated_bases:
            gas_proton = gas("[H+]")
            proton_solvation = self.ion_solvation["[H+]"].value
            proton = gas_proton["gibbs_eV"] * EV_TO_KCAL_MOL + proton_solvation
            rt_ln10 = R_KCAL * self.temperature_K * log(10)
            if rule := self.deprotonated_acids.get(species):
                shift = rt_ln10 * rule.value - proton
                enthalpy_shift = shift
                convention = "retain aqueous conjugate acid entropy"
                if rule.enthalpy_kcal_mol is not None:
                    proton_h = (
                        gas_proton["enthalpy_eV"] * EV_TO_KCAL_MOL + proton_solvation
                    )
                    enthalpy_shift = rule.enthalpy_kcal_mol - proton_h
                    convention = "use aqueous ionization enthalpy and pKa"
                return (
                    water(rule.parent),
                    rule.parent,
                    _correction(
                        "derived_from_conjugate_acid",
                        rule,
                        shift,
                        enthalpy_shift,
                        convention,
                    ),
                )
            rule = self.protonated_bases[species]
            parent = water(rule.parent)
            shift = proton - rt_ln10 * rule.value
            try:
                raw = water(species)
            except MolecularCalculationError as exc:
                correction = _correction(
                    "base_pka",
                    rule,
                    shift,
                    shift,
                    "retain aqueous neutral base entropy",
                )
                correction["raw_species_failure"] = {
                    "species": species,
                    **exc.to_dict(),
                }
                return parent, rule.parent, correction
            shift += (parent["gibbs_eV"] - raw["gibbs_eV"]) * EV_TO_KCAL_MOL
            return (
                raw,
                species,
                _correction(
                    "base_pka", rule, shift, shift, "retain CPCM species entropy"
                ),
            )

        raw = water(species)
        correction = None
        if rule := self.ion_solvation.get(species):
            shift = (
                rule.value
                + (gas(species)["gibbs_eV"] - raw["gibbs_eV"]) * EV_TO_KCAL_MOL
            )
            correction = _correction(
                "ion_solvation", rule, shift, shift, "retain CPCM species entropy"
            )
        elif rule := self.strong_acids.get(species):
            bare = gas(species)
            dg = rule.value + (bare["gibbs_eV"] - raw["gibbs_eV"]) * EV_TO_KCAL_MOL
            dh = (
                rule.enthalpy_kcal_mol
                + (bare["enthalpy_eV"] - raw["enthalpy_eV"]) * EV_TO_KCAL_MOL
            )
            correction = _correction(
                "strong_acid_dissociation",
                rule,
                dg,
                dh,
                "use experimental enthalpy and Gibbs energy",
            )
        return raw, species, correction

    def molecular_values(self, species, phase, gas, water):
        """Apply both measured cycles to a copy and preserve every raw molecular record."""
        corrections = []
        if rule := self.gas_acidities.get(species):
            corrections.append(self._gas_acidity(species, rule, gas))
        if phase == "gas":
            raw, base_species = gas(species), species
        else:
            raw, base_species, entry = self._aqueous_values(species, gas, water)
            if entry is not None:
                corrections.append(entry)
        if not corrections:
            return raw
        derived = base_species != species or (phase != "gas" and species == "[H+]")
        enthalpy = (
            raw["enthalpy_eV"]
            + sum(c["delta_h_kcal_mol"] for c in corrections) / EV_TO_KCAL_MOL
        )
        gibbs = (
            raw["gibbs_eV"]
            + sum(c["delta_g_kcal_mol"] for c in corrections) / EV_TO_KCAL_MOL
        )
        return {
            **({} if derived else raw),
            "energy_eV": None if derived else raw["energy_eV"],
            "enthalpy_eV": enthalpy,
            "gibbs_eV": gibbs,
            "raw_reference_gibbs_eV": raw["gibbs_eV"],
            "raw_reference_enthalpy_eV": raw["enthalpy_eV"],
            "entropy_eV_per_K": (enthalpy - gibbs) / self.temperature_K,
            "reference_base_species": base_species,
            "corrections": corrections,
        }
