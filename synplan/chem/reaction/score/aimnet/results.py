"""Thermochemical sums and copied route annotations."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from math import fsum, isfinite
from typing import TYPE_CHECKING

from .molecular import EV_TO_KCAL_MOL

if TYPE_CHECKING:
    from synplan.chem.reaction.routes import Route

ASSESSMENT_VERSION = 1


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
            "successful_variants": sum(r.get("successful_variants", 0) for r in rows),
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
