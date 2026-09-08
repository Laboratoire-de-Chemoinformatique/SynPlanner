"""Preserve source stereo alongside the explicitly connectivity-only CGR graph.

Snapshots are checked against each deconvolved graph before use. A changed CGR
cannot silently reuse a stereo assignment from a different reaction.
"""

import json
from copy import deepcopy
from hashlib import sha256

from synplan.chem.stereo import parse_smiles_preserving_stereo


def graph_digest(cgr):
    atoms = sorted(
        (
            n,
            a.atomic_number,
            a.isotope,
            a.charge,
            a.p_charge,
            a.is_radical,
            a.p_is_radical,
        )
        for n, a in cgr.atoms()
    )
    bonds = sorted(
        (min(n, m), max(n, m), b.order, b.p_order) for n, m, b in cgr.bonds()
    )
    return sha256(json.dumps((atoms, bonds)).encode()).hexdigest()


def snapshot(reaction, cgr):
    return {
        "reaction": format(reaction, "m"),
        "graph": graph_digest(cgr),
        "metadata": deepcopy(reaction.meta),
    }


def restore(record, cgr):
    if record["graph"] != graph_digest(cgr):
        raise ValueError(
            "RouteCGR graph changed: stereo requires reassessment from the source reaction"
        )
    reaction = parse_smiles_preserving_stereo(record["reaction"])
    reaction.meta.update(deepcopy(record["metadata"]))
    return reaction


def remap_source_cgr(cgr, mapping, *, copy=True):
    source = getattr(cgr, "_stereo_source", None)
    result = cgr.remap(mapping, copy=copy)
    if result is None:
        result = cgr
    if source is not None:
        source = source.copy()
        for molecule in source.molecules():
            molecule.remap({n: m for n, m in mapping.items() if molecule.has_atom(n)})
        source.flush_cache()
        result._stereo_source = source
    return result
