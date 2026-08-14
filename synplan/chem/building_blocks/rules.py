"""Protecting-group rules owned by the building-block preparation feature."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

from chython import smarts

from .config import DeprotectionPolicy


class ProtectiveRule(NamedTuple):
    """One atom-preserving deprotection transformation."""

    query: object
    keep_atoms: tuple[int, ...]
    add_atoms: tuple[tuple[int, str, int], ...]
    protected_smiles: str
    cleaved_smiles: str
    decoys: tuple[str, ...]
    policy: DeprotectionPolicy
    decoy_scope: str


def protective_rules_path() -> Path:
    """Return the bundled taxonomy path used by preparation runs."""
    resource = files("synplan.chem.building_blocks").joinpath(
        "data/protective_rules.tsv"
    )
    return Path(str(resource))


@lru_cache(maxsize=1)
def load_protective_rules() -> dict[str, ProtectiveRule]:
    """Load and parse the bundled, reviewed protective-group taxonomy."""
    rules_path = protective_rules_path()
    rules: dict[str, ProtectiveRule] = {}
    with rules_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "name",
            "smarts",
            "keep_atoms",
            "add_atoms",
            "protected_smiles",
            "cleaved_smiles",
            "decoys",
            "policy",
            "decoy_scope",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("protective_rules.tsv has an unexpected schema")
        for row in reader:
            name = row["name"]
            if name in rules:
                raise ValueError(f"duplicate protective rule name: {name}")
            policy = row["policy"]
            if policy not in {"conservative", "aggressive"}:
                raise ValueError(f"invalid policy for protective rule {name}: {policy}")
            rules[name] = ProtectiveRule(
                query=smarts(row["smarts"]),
                keep_atoms=tuple(json.loads(row["keep_atoms"])),
                add_atoms=tuple(tuple(item) for item in json.loads(row["add_atoms"])),
                protected_smiles=row["protected_smiles"],
                cleaved_smiles=row["cleaved_smiles"],
                decoys=tuple(json.loads(row["decoys"])),
                policy=policy,
                decoy_scope=row["decoy_scope"],
            )
    conservative = sum(rule.policy == "conservative" for rule in rules.values())
    aggressive = sum(rule.policy == "aggressive" for rule in rules.values())
    if (conservative, aggressive) != (84, 11):
        raise ValueError(
            "protective rule taxonomy must contain 84 conservative and "
            f"11 aggressive rules, got {conservative} and {aggressive}"
        )
    return rules


protective_rules = load_protective_rules()

__all__ = [
    "ProtectiveRule",
    "load_protective_rules",
    "protective_rules",
    "protective_rules_path",
]
