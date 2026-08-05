"""Compatibility tests for the canonical molecule API."""

from synplan.chem import molecule
from synplan.chem.precursor import (
    Precursor as LegacyPrecursor,
)
from synplan.chem.precursor import (
    compose_precursors as legacy_compose_precursors,
)
from synplan.chem.utils import (
    mol_from_smiles as legacy_mol_from_smiles,
)
from synplan.chem.utils import (
    safe_canonicalization as legacy_safe_canonicalization,
)
from synplan.chem.utils import (
    standardize_building_blocks as legacy_standardize_building_blocks,
)
from synplan.chem.utils import (
    standardize_sdf_text as legacy_standardize_sdf_text,
)
from synplan.chem.utils import (
    standardize_smiles_batch as legacy_standardize_smiles_batch,
)
from synplan.chem.utils import (
    unite_molecules as legacy_unite_molecules,
)


def test_precursor_compatibility_exports_are_identical():
    assert LegacyPrecursor is molecule.Precursor
    assert legacy_compose_precursors is molecule.compose_precursors


def test_utils_compatibility_exports_are_identical():
    assert legacy_mol_from_smiles is molecule.mol_from_smiles
    assert legacy_safe_canonicalization is molecule.safe_canonicalization
    assert legacy_unite_molecules is molecule.unite_molecules
    assert legacy_standardize_building_blocks is molecule.standardize_building_blocks
    assert legacy_standardize_sdf_text is molecule.standardize_sdf_text
    assert legacy_standardize_smiles_batch is molecule.standardize_smiles_batch
