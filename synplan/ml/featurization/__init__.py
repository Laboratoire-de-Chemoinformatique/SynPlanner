"""Molecular NumPy features and optional Torch/PyG featurization."""

from synplan.ml.featurization.molecules import (
    MENDEL_INFO,
    atom_features,
    atom_to_vector,
    bond_features,
    bonds_to_vector,
    mol_to_matrix,
    mol_to_numpy,
    mol_to_pyg,
    molecule_features,
)

__all__ = [
    "MENDEL_INFO",
    "atom_features",
    "atom_to_vector",
    "bond_features",
    "bonds_to_vector",
    "mol_to_matrix",
    "mol_to_numpy",
    "mol_to_pyg",
    "molecule_features",
    "query_cgr_graph_from_rule_query",
    "query_cgr_graphs_from_smarts",
    "query_cgr_to_pyg",
    "rule_fingerprints_from_smarts",
]


def __getattr__(name):
    if name == "rule_fingerprints_from_smarts":
        from . import fingerprints

        return getattr(fingerprints, name)
    if name in {
        "query_cgr_graph_from_rule_query",
        "query_cgr_graphs_from_smarts",
        "query_cgr_to_pyg",
    }:
        from . import rules

        return getattr(rules, name)
    raise AttributeError(name)
