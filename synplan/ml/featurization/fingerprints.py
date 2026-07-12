"""Torch tensorizers turning reaction rules into MHN ranking fingerprints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import torch
from chython import smarts
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.reaction.rules.representation import (
    RuleFingerprintConfig,
    query_cgr_morgan_fingerprint,
    query_reaction_atom_labels,
    reaction_query_to_reaction,
    rule_fingerprint_digest,
)
from synplan.ml.featurization.cache import (
    cache_get,
    cache_set,
)

_RULE_FINGERPRINT_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()


def _side_fingerprint(
    molecules: Sequence[MoleculeContainer],
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    """Max-pool fragment fingerprints for one side of a reaction rule."""
    if not molecules:
        return torch.zeros(fingerprint_config.fp_size, dtype=torch.float)

    fingerprints = [
        torch.as_tensor(
            molecule.morgan_fingerprint(
                min_radius=fingerprint_config.min_radius,
                max_radius=fingerprint_config.max_radius,
                length=fingerprint_config.fp_size,
                number_active_bits=fingerprint_config.active_bits,
            ),
            dtype=torch.float,
        )
        for molecule in molecules
    ]
    return torch.stack(fingerprints).amax(dim=0)


def _legacy_rule_fingerprint(
    rule_query: ReactionContainer,
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    reaction = reaction_query_to_reaction(rule_query)
    target = _side_fingerprint(reaction.reactants, fingerprint_config)
    precursors = _side_fingerprint(reaction.products, fingerprint_config)
    return target - 0.5 * precursors


def _query_cgr_rule_fingerprint(
    rule_query: ReactionContainer,
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    return torch.as_tensor(
        query_cgr_morgan_fingerprint(
            rule_query.compose(),
            atom_labels=query_reaction_atom_labels(rule_query),
            min_radius=fingerprint_config.min_radius,
            max_radius=fingerprint_config.max_radius,
            length=fingerprint_config.fp_size,
            number_active_bits=fingerprint_config.active_bits,
        ),
        dtype=torch.float,
    )


def _mhnreact_rdkit_fragment_fingerprint(
    fragment_smarts: str,
    fingerprint_config: RuleFingerprintConfig,
    *,
    side: str,
) -> torch.Tensor:
    """Return the original MHNreact RDKit RDK fingerprint for one SMARTS part."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdmolops import FastFindRings

    mol = Chem.MolFromSmarts(str(fragment_smarts), mergeHs=False)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse {side} SMARTS fragment: {fragment_smarts!r}"
        )
    Chem.SanitizeMol(mol, catchErrors=True)
    FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)
    bit_vector = Chem.RDKFingerprint(mol, fpSize=fingerprint_config.fp_size, maxPath=6)
    array = np.zeros((fingerprint_config.fp_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return torch.from_numpy(array)


def _mhnreact_rdkit_side_fingerprint(
    side_smarts: str,
    fingerprint_config: RuleFingerprintConfig,
    *,
    side: str,
) -> torch.Tensor:
    """Max-pool original MHNreact RDKit fingerprints over one template side."""
    fragments = [fragment for fragment in str(side_smarts).split(".") if fragment]
    if not fragments:
        return torch.zeros(fingerprint_config.fp_size, dtype=torch.float)
    fingerprints = [
        _mhnreact_rdkit_fragment_fingerprint(fragment, fingerprint_config, side=side)
        for fragment in fragments
    ]
    return torch.stack(fingerprints).amax(dim=0)


def _mhnreact_rdkit_rule_fingerprint(
    rule_smarts_text: str, fingerprint_config: RuleFingerprintConfig
) -> torch.Tensor:
    """Original MHNreact encoding for ``product>>reactants`` templates."""
    parts = str(rule_smarts_text).split(">")
    if len(parts) != 3:
        raise ValueError(
            "mhnreact_rdkit expects exactly three reaction SMARTS fields "
            "in product>>reactants or product>reagents>reactants form"
        )
    product_side, _reagent_side, reactant_side = parts
    if not product_side or not reactant_side:
        raise ValueError(
            "mhnreact_rdkit requires non-empty product and reactant SMARTS sides"
        )
    product = _mhnreact_rdkit_side_fingerprint(
        product_side, fingerprint_config, side="product"
    )
    reactants = _mhnreact_rdkit_side_fingerprint(
        reactant_side, fingerprint_config, side="reactant"
    )
    return product - 0.5 * reactants


def rule_fingerprints_from_smarts(
    rule_smarts: Sequence[str],
    fingerprint_config: RuleFingerprintConfig | None = None,
) -> torch.Tensor:
    """Build ordered rule fingerprints for retrospective reaction SMARTS.

    ``legacy`` uses the Chython side delta, ``target - 0.5 * precursors``,
    after converting query rules to ordinary reaction containers.
    ``mhnreact_rdkit`` reproduces original MHNreact's RDKit SMARTS template
    encoding, ``left_side - 0.5 * right_side`` with RDK path fingerprints.
    ``query_cgr`` fingerprints Chython ``QueryCGRContainer`` objects with
    original query-side atom labels so constraints such as hydrogen count and
    ring size are retained.
    """
    fingerprint_config = fingerprint_config or RuleFingerprintConfig()
    rules = tuple(rule_smarts)
    fingerprint_digest = rule_fingerprint_digest(rules, fingerprint_config)
    cached = cache_get(_RULE_FINGERPRINT_CACHE, fingerprint_digest)
    if cached is not None:
        return cached

    fingerprints = []
    for index, rule_smarts_text in enumerate(rules):
        try:
            if fingerprint_config.fp_type == "mhnreact_rdkit":
                fingerprint = _mhnreact_rdkit_rule_fingerprint(
                    rule_smarts_text, fingerprint_config
                )
            else:
                rule_query = smarts(rule_smarts_text)
                if fingerprint_config.fp_type == "legacy":
                    fingerprint = _legacy_rule_fingerprint(
                        rule_query, fingerprint_config
                    )
                else:
                    fingerprint = _query_cgr_rule_fingerprint(
                        rule_query, fingerprint_config
                    )
            fingerprints.append(fingerprint)
        except Exception as err:
            raise ValueError(
                f"Failed to fingerprint reaction rule at index {index}:\n"
                f"  SMARTS: {rule_smarts_text}\n"
                f"  error: {type(err).__name__}: {err}"
            ) from err

    tensor = (
        torch.stack(fingerprints)
        if fingerprints
        else torch.empty((0, fingerprint_config.fp_size), dtype=torch.float)
    )
    cache_set(_RULE_FINGERPRINT_CACHE, fingerprint_digest, tensor)
    return tensor


__all__ = [
    "rule_fingerprints_from_smarts",
]
