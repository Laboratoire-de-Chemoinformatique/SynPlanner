"""Torch featurization layer turning chython rules and molecules into tensors and PyG ``Data``."""

from synplan.ml.featurization.molecules import (
    MENDEL_INFO,
    atom_to_vector,
    bonds_to_vector,
    mol_to_matrix,
    mol_to_numpy,
    mol_to_pyg,
)

__all__ = [
    "MENDEL_INFO",
    "atom_to_vector",
    "bonds_to_vector",
    "mol_to_matrix",
    "mol_to_numpy",
    "mol_to_pyg",
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
