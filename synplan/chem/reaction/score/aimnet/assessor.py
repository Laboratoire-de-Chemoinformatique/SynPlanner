"""Post-planning orchestration of balanced reaction and copied route assessments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from synplan.chem.utils import mapped_smiles

from .cache import MolecularCache, MolecularCalculator
from .config import AIMNetConfig, MolecularCalculationError, normalize_config
from .molecular import PROTON_REFERENCE, AIMNetMolecularCalculator
from .phase import (
    apply_phase_conventions,
    net_coefficients,
    separates_ions,
)
from .preparation import prepare_reaction, reaction_variants
from .results import ASSESSMENT_VERSION, annotate_routes, reaction_thermochemistry

if TYPE_CHECKING:
    from chython.containers import ReactionContainer

    from synplan.chem.reaction.routes import Route


class AIMNetReactionAssessor:
    """Annotate copied Routes, retaining every unique missing-stereo combination.

    ``cache_path=None`` keeps successful molecules in memory only. A file cache
    is atomically checkpointed after each new successful species, shared across
    sequential runs (concurrent writers are not supported). Expected chemical
    failures are reported; dependencies, checkpoint and unexpected errors raise.
    ``calculator`` is an optional adapter for alternate execution or testing.
    """

    def __init__(
        self,
        config: AIMNetConfig | None = None,
        cache_path: str | Path | None = "aimnet_cache.json",
        *,
        calculator: MolecularCalculator | None = None,
        gas_calculator: MolecularCalculator | None = None,
    ):
        self.config = normalize_config(config or AIMNetConfig())
        self.calculator = calculator or AIMNetMolecularCalculator(self.config)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache = MolecularCache(self.config, self.cache_path)
        self.gas = None
        if self.config.phase == "cpcm_water" and self.config.also_calculate_gas:
            gas_config = self.config.model_copy(
                update={
                    "phase": "gas",
                    "also_calculate_gas": False,
                    "cpcm_checkpoint": None,
                    "cpcm_sha256": None,
                    "solvent_concentration_mol_l": None,
                }
            )
            gas_path = self.cache_path
            if gas_path is not None and not self._cache.disk_backed:
                gas_path = gas_path.with_name(gas_path.stem + ".gas" + gas_path.suffix)
            if gas_calculator is None and isinstance(
                self.calculator, AIMNetMolecularCalculator
            ):
                gas_calculator = AIMNetMolecularCalculator(
                    gas_config, generator=self.calculator.generator
                )
            self.gas = AIMNetReactionAssessor(
                gas_config, gas_path, calculator=gas_calculator
            )
        self.last_stats = {}
        self.completed_reactions: list[dict] = []

    @property
    def cache(self) -> dict:
        """Molecular cache data, also used to prime isolated workers."""
        return self._cache.data

    @property
    def identity(self) -> dict | None:
        return self._cache.identity

    def _molecule(self, smiles: str) -> dict:
        return self._cache.get(smiles, self.calculator, self.last_stats)

    def run_identity(self):
        identity = self.calculator.identity()
        if self.gas:
            return {
                "primary": identity,
                "gas": self.gas.calculator.identity(),
            }
        return identity

    def _reaction(self, reaction: ReactionContainer) -> dict:
        result = {
            "version": ASSESSMENT_VERSION,
            "status": "skipped",
            "original_reaction": mapped_smiles(reaction),
            "balanced_reaction": None,
            "variants": [],
            "molecules": {},
        }
        try:
            prepared, reactants, products = prepare_reaction(
                reaction,
                self.config.min_confidence,
                infer_balance=self.config.balance_policy == "infer",
            )
        except MolecularCalculationError as exc:
            result.update(exc.reaction_context)
            result["failure"] = exc.to_dict()
            result["status"] = "skipped" if exc.category == "filtered_out" else "failed"
            return result
        result.update(prepared)
        if (
            self.config.phase == "gas"
            and self.config.withhold_gas_ion_separation
            and separates_ions(net_coefficients(reactants, products))
        ):
            result["failure"] = {
                "category": "filtered_out",
                "stage": "publication",
                "reason": "gas_charge_separation",
                "detail": "The equation changes the charge-squared sum of separated ions.",
            }
            return result
        assignments = (
            reaction_variants(reactants, products)
            if self.config.stereo_policy == "enumerate"
            else [(reactants, products)]
        )
        for left, right in assignments:
            variant = {
                "id": len(result["variants"]) + 1,
                "reaction": ".".join(left) + ">>" + ".".join(right),
                "reactants": left,
                "products": right,
                "status": "ok",
                "delta_e": None,
                "delta_h": None,
                "delta_g": None,
                "units": "kcal/mol",
            }
            failures = []
            for smiles in dict.fromkeys([*left, *right]):
                if smiles in result["molecules"]:
                    continue
                try:
                    result["molecules"][smiles] = deepcopy(self._molecule(smiles))
                except MolecularCalculationError as exc:
                    failures.append(
                        {
                            "smiles": smiles,
                            **exc.to_dict(),
                            "conformer_failures": getattr(
                                exc, "conformer_failures", []
                            ),
                        }
                    )
            if failures:
                variant.update(status="failed", failures=failures)
            else:
                variant.update(
                    reaction_thermochemistry(left, right, result["molecules"])
                )
                apply_phase_conventions(variant, self.config)
            result["variants"].append(variant)
        if not result["variants"]:
            result["status"] = "failed"
            result["failure"] = {
                "category": "errored",
                "stage": "stereo",
                "reason": "no_variants",
                "detail": "No stereochemical assignments could be enumerated.",
            }
            return result
        successful = sum(v["status"] == "ok" for v in result["variants"])
        result.update(
            status="ok"
            if successful == len(result["variants"])
            else ("partial" if successful else "failed"),
            total_variants=len(result["variants"]),
            successful_variants=successful,
        )
        return result

    def assess_reactions(
        self,
        reactions: Iterable[ReactionContainer],
        *,
        silent: bool = True,
        on_reaction: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        """Return aligned, JSON-compatible annotations; failures are never zeros."""
        started = perf_counter()
        self.last_stats = {
            "cache_hits": 0,
            "cache_invalidations": 0,
            "molecular_calculations": 0,
            "molecular_failures": 0,
            "calculation_seconds": 0.0,
        }
        self._cache.begin_assessment()
        if self.gas:
            self.gas._cache.begin_assessment()
            self.gas.last_stats = self.last_stats
        self.completed_reactions = []
        results = self.completed_reactions
        for index, reaction in enumerate(
            tqdm(
                reactions,
                desc="AIMNet thermochemistry",
                unit="reaction",
                disable=silent,
            )
        ):
            result = self._reaction(reaction)
            if self.config.also_calculate_gas:
                assert self.gas is not None
                alternate = self.gas._reaction(reaction)
                gas_variants = {
                    v["reaction"]: v
                    for v in alternate["variants"]
                    if v["status"] == "ok"
                }
                for variant in result["variants"]:
                    gas = gas_variants.get(variant["reaction"])
                    if gas is not None and variant["status"] == "ok":
                        variant["delta_delta_g_solv_kcal_mol"] = (
                            variant["delta_g"] - gas["delta_g"]
                        )
                alternate["calculation"] = self.gas._calculation_provenance()
                result.update(
                    version=2, phase="cpcm_water", additional_phases={"gas": alternate}
                )
            result["calculation"] = self._calculation_provenance()
            results.append(result)
            if on_reaction is not None:
                on_reaction(index, result)
        self.last_stats.update(
            reaction_occurrences=len(results),
            unique_species=len(
                self._cache.seen_species.union(
                    self.gas._cache.seen_species if self.gas else ()
                )
            ),
            successful_reactions=sum(r["status"] == "ok" for r in results),
            skipped_reactions=sum(r["status"] == "skipped" for r in results),
            total_variants=sum(len(r["variants"]) for r in results),
            successful_variants=sum(r.get("successful_variants", 0) for r in results),
            elapsed_seconds=perf_counter() - started,
        )
        for result in results:
            result["calculation"] = self._calculation_provenance()
        if not silent:
            tqdm.write(
                f"AIMNet: {self.last_stats['successful_variants']}/{self.last_stats['total_variants']} variants; {self.last_stats['molecular_calculations']} molecular calculations, {self.last_stats['cache_hits']} cache hits."
            )
        return results

    def _calculation_provenance(self) -> dict:
        return {
            "identity": deepcopy(self.identity),
            **(
                {"gas_reference_calculator": deepcopy(self.gas.identity)}
                if self.gas
                else {}
            ),
            "config": self.config.model_dump(mode="json"),
            "method": (
                "gas_phase_single_conformer_RRHO"
                if self.config.phase == "gas"
                else "cpcm_water_single_conformer_RRHO"
            ),
            "reference_conventions": {
                "proton_electronic_energy": "0 eV",
                "proton_reference_version": PROTON_REFERENCE,
            },
        }

    def assess(
        self,
        routes: Iterable[Route],
        *,
        silent: bool = True,
        on_reaction: Callable[[int, dict], None] | None = None,
    ) -> list[Route]:
        """Return annotated copies in input order, preserving other assessments."""
        routes = list(routes)
        annotations = iter(
            self.assess_reactions(
                (step.reaction for route in routes for step in route.steps),
                silent=silent,
                on_reaction=on_reaction,
            )
        )
        assessed = self.annotate_routes(routes, annotations)
        self.last_stats["routes"] = len(routes)
        return assessed

    @staticmethod
    def annotate_routes(
        routes: Iterable[Route], annotations: Iterable[dict]
    ) -> list[Route]:
        """Attach aligned results to route copies, also usable for checkpoints."""
        return annotate_routes(routes, annotations)
