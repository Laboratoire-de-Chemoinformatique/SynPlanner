"""Post-planning AIMNet2 gas-phase thermochemistry of balanced reaction variants.

The public assessor follows ForwardReactionAssessor's copy-and-annotate API.
Neither planning, route order, nor the original reaction containers are changed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import replace
from math import fsum, isfinite
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

from tqdm.auto import tqdm

from synplan.chem.reaction.score.chemistry import (
    canonical_species,
    is_proton,
    parse_species,
    prepare_reaction,
    reaction_variants,
)
from synplan.chem.reaction.score.config import AIMNetConfig, MolecularCalculationError
from synplan.chem.reaction.score.molecular import (
    EV_TO_KCAL_MOL,
    PROTON_REFERENCE,
    AIMNetMolecularCalculator,
)
from synplan.chem.utils import mapped_smiles
from synplan.utils.files import read_json, write_json_atomic

if TYPE_CHECKING:
    from chython.containers import ReactionContainer

    from synplan.chem.reaction.routes import Route

__all__ = ["AIMNetConfig", "AIMNetReactionAssessor"]

ASSESSMENT_VERSION = 1
CACHE_VERSION = 1


class MolecularCalculator(Protocol):
    """Injectable adapter; identity must identify model bytes and runtime versions."""

    def identity(self) -> dict: ...

    def calculate(self, smiles: str) -> dict: ...


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def reaction_thermochemistry(
    reactants: list[str], products: list[str], molecules: dict
) -> dict:
    """Stoichiometric products minus reactants; duplicate species count repeatedly."""
    result = {"units": "kcal/mol"}
    for name, field in (
        ("delta_e", "energy_eV"),
        ("delta_h", "enthalpy_eV"),
        ("delta_g", "gibbs_eV"),
    ):
        value = (
            fsum(
                [
                    *(molecules[s][field] for s in products),
                    *(-molecules[s][field] for s in reactants),
                ]
            )
            * EV_TO_KCAL_MOL
        )
        if not isfinite(value):
            raise ValueError("Molecular calculator returned nonfinite thermochemistry")
        result[name] = value
    return result


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
    ):
        config = config or AIMNetConfig()
        symmetry = {}
        for text, number in config.symmetry_numbers.items():
            key = canonical_species(parse_species(text))
            if key in symmetry and symmetry[key] != number:
                raise ValueError(f"Conflicting symmetry numbers for {key}")
            symmetry[key] = number
        self.config = config.model_copy(
            update={"symmetry_numbers": symmetry}, deep=True
        )
        self.calculator = calculator or AIMNetMolecularCalculator(self.config)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache: dict[str, Any] = (
            read_json(self.cache_path)
            if self.cache_path is not None and self.cache_path.exists()
            else {"version": CACHE_VERSION, "calculations": {}}
        )
        if self.cache.get("version") != CACHE_VERSION:
            raise ValueError("Unsupported AIMNet cache version")
        self.identity = None
        self._entries = None
        self.last_stats = {}
        self.completed_reactions: list[dict] = []

    def _molecule(self, smiles: str) -> dict:
        if self._entries is None:
            self.identity = {
                "assessment_version": ASSESSMENT_VERSION,
                "config": self.config.model_dump(mode="json"),
                "calculator": self.calculator.identity(),
            }
            key = _fingerprint(self.identity)
            self._entries = self.cache["calculations"].setdefault(
                key, {"identity": self.identity, "molecules": {}}
            )["molecules"]
        self._seen_species.add(smiles)
        if (
            smiles in self._entries
            and is_proton(smiles)
            and self._entries[smiles].get("electronic_energy_reference")
            != PROTON_REFERENCE
        ):
            del self._entries[smiles]
        if smiles in self._entries:
            self.last_stats["cache_hits"] += 1
            return self._entries[smiles]
        if smiles in self._failures:
            raise self._failures[smiles]
        self.last_stats["molecular_calculations"] += 1
        started = perf_counter()
        try:
            result = self.calculator.calculate(smiles)
            if not all(
                isfinite(result[k]) for k in ("energy_eV", "enthalpy_eV", "gibbs_eV")
            ):
                raise ValueError("Molecular calculator returned nonfinite E/H/G")
        except MolecularCalculationError as exc:
            self._failures[smiles] = exc
            self.last_stats["molecular_failures"] += 1
            raise
        finally:
            self.last_stats["calculation_seconds"] += perf_counter() - started
        self._entries[smiles] = deepcopy(result)
        if self.cache_path is not None:
            write_json_atomic(self.cache_path, self.cache)
        return self._entries[smiles]

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
                reaction, self.config.min_confidence
            )
        except MolecularCalculationError as exc:
            result.update(exc.reaction_context)
            result["failure"] = exc.to_dict()
            result["status"] = "skipped" if exc.category == "filtered_out" else "failed"
            return result
        result.update(prepared)
        for left, right in reaction_variants(reactants, products):
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
            "molecular_calculations": 0,
            "molecular_failures": 0,
            "calculation_seconds": 0.0,
        }
        self._seen_species: set[str] = set()
        self._failures: dict[str, MolecularCalculationError] = {}
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
            result["calculation"] = self._calculation_provenance()
            results.append(result)
            if on_reaction is not None:
                on_reaction(index, result)
        self.last_stats.update(
            reaction_occurrences=len(results),
            unique_species=len(self._seen_species),
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
            "config": self.config.model_dump(mode="json"),
            "method": "gas_phase_single_conformer_RRHO",
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
        annotations = iter(annotations)
        assessed = []
        for route in routes:
            steps = tuple(
                replace(step, thermochemistry=next(annotations)) for step in route.steps
            )
            rows = [step.thermochemistry for step in steps]
            summary = {
                "version": ASSESSMENT_VERSION,
                "total_steps": len(steps),
                "successful_steps": sum(r["status"] == "ok" for r in rows),
                "skipped_steps": sum(r["status"] == "skipped" for r in rows),
                "failed_steps": sum(r["status"] == "failed" for r in rows),
                "partial_steps": sum(r["status"] == "partial" for r in rows),
                "total_variants": sum(len(r["variants"]) for r in rows),
                "successful_variants": sum(
                    r.get("successful_variants", 0) for r in rows
                ),
                "calculation": deepcopy(rows[0]["calculation"]),
            }
            assessed.append(
                replace(
                    route,
                    steps=steps,
                    assessment={**deepcopy(route.assessment or {}), "aimnet": summary},
                )
            )
        return assessed
