"""Shared augmented forward prediction; chemistry identity is stereo-free InChIKey.

Rooted input augmentation and best-rank/reciprocal-rank ordering are adapted from
Microsoft RetroChimera v1.3.0 (bf5ec59eec9ef32911d5167bf7528e3648622be1).
Copyright (c) Microsoft Corporation and contributors. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
Modifications: deterministic augmentation, chython stereo-free deduplication,
representative selection, and batching independent of neural architecture.
"""

import hashlib
import math
import random
from functools import reduce
from operator import or_

from chython import inchi_key, smiles
from chython.containers import MoleculeContainer
from chython.exceptions import ValenceError
from rdkit import Chem

PIPELINE_VERSION = 2


def _molecule(value):
    # chython 1.108 can invert stereo at a later component's root. Parsing
    # components separately and using its union preserves their local ordering.
    if "@" in value and "." in value:
        return reduce(or_, (smiles(part) for part in value.split(".")))
    return smiles(value)


def canonical_reactants(value):
    """Unmap using chython; preserve stereo, isotopes and tautomer form."""
    if not value.strip() or ">" in value:
        raise ValueError("Supply nonempty reactants-only SMILES separated by dots")
    molecule = Chem.MolFromSmiles(format(_molecule(value), ""))
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f"Invalid reactant SMILES: {value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def candidate(value):
    if ">" in value:
        return None, None
    try:
        molecule = _molecule(value)
        if not isinstance(molecule, MoleculeContainer) or not molecule:
            return None, None
        molecule.canonicalize(fix_tautomers=False)
        return str(molecule), inchi_key(molecule, ignore_stereo=True)
    except (ValueError, ValenceError):
        return None, None


def augment(reactants, count=10, seed=0):
    """Distinct rooted spellings, canonical first, with a local input-specific RNG."""
    variants = []
    for component in reactants.split("."):
        mol = Chem.MolFromSmiles(component)
        variants.append(
            list(
                dict.fromkeys(
                    Chem.MolToSmiles(
                        mol, rootedAtAtom=root, canonical=True, isomericSmiles=True
                    )
                    for root in [-1, *range(mol.GetNumAtoms())]
                )
            )
        )
    total = math.prod(map(len, variants))
    rng = random.Random(
        int.from_bytes(hashlib.sha256(f"{seed}\0{reactants}".encode()).digest(), "big")
    )
    indices, seen, result = set(), set(), []
    # Bound work for highly symmetric mixtures without enumerating a huge product.
    for attempt in range(min(total, max(200, count * 20))):
        index = 0 if attempt == 0 else rng.randrange(total)
        while index in indices:
            index = rng.randrange(total)
        indices.add(index)
        components = []
        for options in variants:
            index, offset = divmod(index, len(options))
            components.append(options[offset])
        value = ".".join(sorted(components))
        if value not in seen:
            seen.add(value)
            result.append(value)
            if len(result) == count:
                break
    return result


def aggregate(reactants, rows, num_results=10):
    """Rank unique identities; one vote per augmentation, never per stereoisomer."""
    identities = {}
    for aug_index, row in enumerate(rows):
        seen = set()
        for beam_index, (raw, likelihood) in enumerate(
            zip(row["predictions"], row["log_likelihoods"], strict=True)
        ):
            canonical, key = candidate(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            occurrence = (rank, aug_index, beam_index)
            if key not in identities:
                identities[key] = {
                    "representative": occurrence,
                    "canonical": canonical,
                    "likelihood": likelihood,
                    "support": 0.0,
                }
            entry = identities[key]
            entry["support"] += 1.0 / rank
            if occurrence < entry["representative"]:
                entry.update(
                    representative=occurrence,
                    canonical=canonical,
                    likelihood=likelihood,
                )
    ordered = sorted(
        identities,
        key=lambda key: (
            identities[key]["representative"][0],
            -identities[key]["support"],
        ),
    )[:num_results]
    entries = [identities[key] for key in ordered]
    return {
        "reactants": reactants,
        "predictions": [entry["canonical"] for entry in entries],
        "canonical_products": [entry["canonical"] for entry in entries],
        "product_inchikeys": ordered,
        "log_likelihoods": [entry["likelihood"] for entry in entries],
        "ranking_scores": [
            entry["support"] - 1e8 * (entry["representative"][0] - 1)
            for entry in entries
        ],
        "augmentation_count": len(rows),
    }


def predict_augmented(
    model, reactants, beam_width=20, batch_size=16, *, num_results=10
):
    if batch_size < 1 or num_results < 1 or beam_width < 1:
        raise ValueError("Batch size, result count and beam width must be positive")
    results = []
    for start in range(0, len(reactants), batch_size):
        inputs = [
            canonical_reactants(value)
            for value in reactants[start : start + batch_size]
        ]
        augmented = [
            augment(value, model.augmentation_size, model.seed) for value in inputs
        ]
        flat = [value for variants in augmented for value in variants]
        rows = []
        # A single reaction's augmentations share an encoder/beam batch.
        chunk = max(batch_size, model.augmentation_size)
        for offset in range(0, len(flat), chunk):
            rows.extend(model._predict_batch(flat[offset : offset + chunk], beam_width))
        offset = 0
        for value, variants in zip(inputs, augmented, strict=True):
            results.append(
                aggregate(value, rows[offset : offset + len(variants)], num_results)
            )
            offset += len(variants)
    return results
