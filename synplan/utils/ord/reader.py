"""Read ORD .pb (Open Reaction Database protobuf) files into ReactionContainer objects.

Uses pre-generated _pb2 stubs from the ORD proto schema — no ord-schema dependency.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from chython import smiles as parse_smiles
from chython.containers import ReactionContainer

logger = logging.getLogger(__name__)

# CompoundIdentifier.IdentifierType enum values (from reaction.proto)
_ID_SMILES = 2
_ID_INCHI = 4
_ID_NAME = 6  # IUPAC_NAME
_ID_NAME2 = 7  # NAME

# Compound.ReactionRole enum values
_ROLE_REACTANT = 1
_ROLE_REAGENT = 2
_ROLE_SOLVENT = 3
_ROLE_CATALYST = 4

# ProductMeasurement.ProductMeasurementType enum values
_MEASURE_YIELD = 3


def _get_smiles(identifiers) -> str | None:
    """Return the first SMILES value from a CompoundIdentifier list."""
    for ident in identifiers:
        if ident.type == _ID_SMILES:
            val = ident.value.strip()
            if val:
                return val
    return None


def _get_name(identifiers) -> str | None:
    """Return the first NAME/IUPAC_NAME from a CompoundIdentifier list."""
    for ident in identifiers:
        if ident.type in (_ID_NAME, _ID_NAME2):
            val = ident.value.strip()
            if val:
                return val
    return None


def _get_yield(measurements) -> float | None:
    """Return the first YIELD percentage from ProductMeasurement list."""
    for m in measurements:
        if m.type == _MEASURE_YIELD and m.HasField("percentage"):
            return m.percentage.value
    return None


def _reaction_to_smiles(reaction) -> tuple[str | None, dict]:
    """Convert an ORD Reaction protobuf to reaction SMILES + metadata dict.

    Returns (reaction_smiles, meta) where reaction_smiles is
    "reactants>reagents>products" or None if no SMILES could be assembled.
    """
    reactant_smiles = []
    reagent_smiles = []
    meta = {}

    names = {"reagents": [], "solvents": [], "catalysts": []}

    for _key, reaction_input in reaction.inputs.items():
        for comp in reaction_input.components:
            smi = _get_smiles(comp.identifiers)
            name = _get_name(comp.identifiers)
            role = comp.reaction_role

            if role == _ROLE_REACTANT:
                if smi:
                    reactant_smiles.append(smi)
            elif role in (_ROLE_REAGENT, _ROLE_SOLVENT, _ROLE_CATALYST):
                if smi:
                    reagent_smiles.append(smi)
                label = {
                    _ROLE_REAGENT: "reagents",
                    _ROLE_SOLVENT: "solvents",
                    _ROLE_CATALYST: "catalysts",
                }[role]
                if name:
                    names[label].append(name)
            else:
                # UNSPECIFIED (0) or other — treat as reactant if has SMILES
                if smi:
                    reactant_smiles.append(smi)

    # A screening record lists everything the plate was read for: the desired
    # product, the isomers competing with it, and an internal standard. Only
    # the first is a product of this equation, and the record says which it is.
    # The flag picks between the isomers of one measurement, so the choice is
    # made inside each outcome: a second outcome that flags nothing is another
    # measurement, not a competitor, and keeps all of its products.
    # The flag is read before the SMILES are: a desired product recorded
    # without a structure must leave the outcome empty rather than promote the
    # sibling it was chosen over.
    products = []
    for outcome in reaction.outcomes:
        desired = [prod for prod in outcome.products if prod.is_desired_product]
        products.extend(
            (prod, smi)
            for prod in (desired or outcome.products)
            if (smi := _get_smiles(prod.identifiers))
        )

    product_smiles = [smi for _, smi in products]
    yields = [
        y for prod, _ in products if (y := _get_yield(prod.measurements)) is not None
    ]

    if not reactant_smiles or not product_smiles:
        return None, {}

    rxn_smi = (
        ".".join(reactant_smiles)
        + ">"
        + ".".join(reagent_smiles)
        + ">"
        + ".".join(product_smiles)
    )

    meta["ord_reaction_id"] = reaction.reaction_id or ""
    if yields:
        meta["ord_yields"] = ";".join(f"{y:.1f}" for y in yields)
    for key, vals in names.items():
        if vals:
            meta[f"ord_{key}"] = ";".join(vals)

    return rxn_smi, meta


def iter_ord_reactions(path: str | Path) -> Iterator[ReactionContainer]:
    """Yield ReactionContainer objects parsed from an ORD .pb Dataset file.

    Reactions without extractable SMILES or with unparseable SMILES are
    skipped with log messages.

    :param path: Path to the ORD Dataset .pb file.
    :yields: ReactionContainer objects with ORD metadata in .meta dict.
    """
    from synplan.utils.ord import dataset_pb2

    path = Path(path)
    dataset = dataset_pb2.Dataset()
    with open(path, "rb") as f:
        dataset.ParseFromString(f.read())

    logger.info("ORD dataset '%s': %d reactions", dataset.name, len(dataset.reactions))

    for reaction in dataset.reactions:
        rxn_smi, meta = _reaction_to_smiles(reaction)
        if rxn_smi is None:
            logger.debug(
                "Skipping reaction %s: could not extract reactant+product SMILES",
                reaction.reaction_id,
            )
            continue

        try:
            rxn = parse_smiles(rxn_smi)
        except Exception as exc:
            logger.warning("chython parse failed for %s: %s", reaction.reaction_id, exc)
            continue

        if not isinstance(rxn, ReactionContainer):
            logger.warning(
                "Not a reaction SMILES for %s: got %s",
                reaction.reaction_id,
                type(rxn).__name__,
            )
            continue

        rxn.meta["init_smiles"] = rxn_smi
        rxn.meta.update(meta)
        yield rxn


def count_ord_reactions(path: str | Path) -> int:
    """Return the number of Reaction entries in an ORD .pb Dataset file."""
    from synplan.utils.ord import dataset_pb2

    dataset = dataset_pb2.Dataset()
    with open(path, "rb") as f:
        dataset.ParseFromString(f.read())
    return len(dataset.reactions)


def convert_ord_to_smiles(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Convert an ORD .pb file to a .smi file compatible with SynPlanner.

    Writes one reaction SMILES per line (tab-separated with metadata).

    :param input_path: Path to the input .pb file.
    :param output_path: Path to the output .smi file.
    :return: Number of reactions written.
    """
    from synplan.utils.files import to_reaction_smiles_record

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for rxn in iter_ord_reactions(input_path):
            out.write(to_reaction_smiles_record(rxn) + "\n")
            count += 1

    logger.info("Wrote %d reactions to %s", count, output_path)
    return count
