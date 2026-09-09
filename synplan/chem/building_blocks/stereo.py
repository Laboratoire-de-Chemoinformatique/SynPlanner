"""Compatible records from the existing immutable InChIKey catalogue."""

from functools import lru_cache

from chython import inchi_key, smiles
from chython.containers import MoleculeContainer

from synplan.chem.building_blocks.core import match_building_blocks
from synplan.chem.mapping import (
    MappingBudgetExceeded,
    backend_preparation,
    bounded_mappings,
    mapping_budget,
)
from synplan.chem.stereo import (
    has_stereo_groups,
    stereo_requirements,
    stereo_sign,
)


@lru_cache(maxsize=8192)
def record_molecule(smiles_text, key):
    with backend_preparation():
        candidate = smiles(smiles_text, strict_stereo=True)
        if not isinstance(candidate, MoleculeContainer):
            raise ValueError("catalogue record must contain a molecule")
        candidate.thiele()
        if inchi_key(candidate) != key:
            raise ValueError("catalogue record SMILES and InChIKey disagree")
    return candidate


def compatible_records(
    molecule,
    catalogue,
    *,
    inchikey=None,
    match_stereo=True,
    max_records=256,
    max_mapping_work=100_000,
    diagnostics=None,
):
    """Return compatible explicit records; unspecified stereo never means racemic.

    Retrieval is indexed. Each explicit candidate is checked at most once; any
    correspondence work is separately bounded. Empty results after a budget
    exception are incomplete, not proof that compatible stock does not exist.
    """
    key = inchikey or inchi_key(molecule)
    if not match_stereo:
        return match_building_blocks(catalogue, key)
    if has_stereo_groups(molecule):
        if diagnostics is not None:
            diagnostics.append(
                {
                    "reason": "relative_or_mixture_stereo",
                    "detail": "exact material semantics require review",
                }
            )
        return ()
    bucket = match_building_blocks(catalogue, key)
    if len(bucket) > max_records:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "reason": "stock_assessment_incomplete",
                    "detail": f"bucket exceeds {max_records} records",
                }
            )
        return ()
    requirements = stereo_requirements(molecule)
    query_smiles = str(molecule)
    compatible = []
    try:
        with mapping_budget(max_mapping_work):
            # Exact representations need no mapping work; check them before alternatives.
            for record in sorted(bucket, key=lambda r: r.smiles != query_smiles):
                if requirements and record.stereo_type not in ("", "absolute"):
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "reason": "relative_or_mixture_stereo",
                                "detail": f"supplier stereo type: {record.stereo_type}",
                                "sources": [dict(s) for s in record.sources],
                            }
                        )
                    continue
                try:
                    candidate = record_molecule(record.smiles, record.inchikey)
                except ValueError as error:
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "reason": "invalid_stock_record",
                                "inchikey": record.inchikey,
                                "detail": str(error),
                            }
                        )
                    continue
                # OR is unresolved absolute identity; AND is a material mixture.
                # Neither satisfies a request for the depicted absolute isomer.
                if (requirements and has_stereo_groups(candidate)) or len(
                    candidate
                ) != len(molecule):
                    continue
                if str(candidate) == query_smiles:
                    compatible.append(record)
                    continue
                for mapping in bounded_mappings(molecule, candidate):
                    try:
                        if all(
                            stereo_sign(candidate, req.remap(mapping)) == req.sign
                            for req in requirements
                        ):
                            compatible.append(record)
                            break
                    except (KeyError, ValueError):
                        continue
    except MappingBudgetExceeded as error:
        if diagnostics is not None:
            diagnostics.append(
                {"reason": "stock_assessment_incomplete", "detail": str(error)}
            )
        # Already verified records remain valid existential matches. Never claim
        # that the incomplete set contains the globally cheapest supplier.
    return tuple(compatible)


def selected_record(record):
    selected = {
        "inchikey": record.inchikey,
        "smiles": record.smiles,
        "price": record.price,
        "basis": "explicit_compatible_catalogue_record",
    }
    if record.sources:
        selected["sources"] = [dict(source) for source in record.sources]
    if record.stereo_type:
        selected["stereo_type"] = record.stereo_type
    return selected


def matches_selected(record, selected):
    """Pin the material and its source declarations across route reassessment."""
    return (
        record.inchikey == selected["inchikey"]
        and record.smiles == selected["smiles"]
        and record.stereo_type == selected.get("stereo_type", "")
        and (
            "sources" not in selected
            or [dict(s) for s in record.sources] == selected["sources"]
        )
    )
