"""Module containing additional functions needed in different reaction data processing
protocols."""

import re
import warnings
from typing import Literal

from chython.containers import (
    MoleculeContainer,
    ReactionContainer,
)
from chython.exceptions import InvalidAromaticRing, MappingError
from chython.files.daylight.tokenize import smarts_tokenize

from synplan.chem.molecule import io as _molecule_io
from synplan.chem.molecule import standardization as _molecule_standardization

# v1.6 compatibility aliases; canonical imports live in synplan.chem.molecule.
standardize_building_blocks = _molecule_io.standardize_building_blocks
standardize_sdf_text = _molecule_io.standardize_sdf_text
standardize_smiles_batch = _molecule_io.standardize_smiles_batch
_clean_molecule = _molecule_standardization._clean_molecule
mol_from_smiles = _molecule_standardization.mol_from_smiles
safe_canonicalization = _molecule_standardization.safe_canonicalization
unite_molecules = _molecule_standardization.unite_molecules

ReactionMappingStatus = Literal["fully_mapped", "partially_mapped", "unmapped"]
AtomMappingCheck = Literal["off", "reject_unmapped", "reject_partial"]

# Strip CXSMILES extension blocks (e.g. ' |^1:4|') before tokenizing: they
# carry no atom-map info and chython's single-side tokenizer rejects them.
_CXSMILES_BLOCK = re.compile(r"\s*\|[^|]*\|\s*")


def reaction_mapping_status(reaction: ReactionContainer) -> ReactionMappingStatus:
    """Classify a parsed reaction's atom-mapping state.

    Returns one of:

    * ``"fully_mapped"`` — every heavy atom is explicitly mapped and at
      least one map number is shared between reactants and products.
    * ``"partially_mapped"`` — some atoms share map numbers, but at least
      one heavy atom was originally bare in the input.
    * ``"unmapped"`` — no atom numbers are shared between reactants and
      products.

    Hydrogens (``atomic_number == 1``) and reagents are skipped.

    .. warning::
        This container-based check has known limitations on
        ``chython.smarts(...)`` output and on containers restored from a
        binary pickle. The SMARTS parser discards the ``_parsed_mapping``
        attribute (``chython/files/daylight/smarts.py``), and the binary
        unpacker resets it to ``None``. Without that attribute we cannot
        distinguish "atom was explicitly mapped" from "atom was bare and
        auto-numbered". For SMARTS rule strings (e.g. from RDKit /
        RDChiral output) use :func:`reaction_string_mapping_status` on
        the **raw text** instead — it inspects the tokenizer output before
        chython drops the trace.

    For SMILES-parsed reactions (the common test-fixture and pipeline
    case) all three states are correctly detected.
    """
    reactant_keys: set[int] = set()
    reactant_atoms: list[object] = []
    for mol in reaction.reactants:
        for n, atom in mol.atoms():
            if atom.atomic_number == 1:
                continue
            reactant_keys.add(n)
            reactant_atoms.append(atom)

    product_keys: set[int] = set()
    product_atoms: list[object] = []
    for mol in reaction.products:
        for n, atom in mol.atoms():
            if atom.atomic_number == 1:
                continue
            product_keys.add(n)
            product_atoms.append(atom)

    if not (reactant_keys & product_keys):
        return "unmapped"

    all_atoms = (*reactant_atoms, *product_atoms)
    has_any_explicit = any(getattr(a, "_parsed_mapping", None) for a in all_atoms)
    if not has_any_explicit:
        # SMARTS-parsed rule or restored from binary pickle — partial
        # detection is not possible from the container alone. Trust the
        # intersection check.
        return "fully_mapped"
    for atom in all_atoms:
        if not getattr(atom, "_parsed_mapping", None):
            return "partially_mapped"
    return "fully_mapped"


def reaction_string_mapping_status(text: str) -> ReactionMappingStatus:
    """Classify a reaction's atom-mapping state from its raw text.

    Uses chython's tokenizer (``smarts_tokenize`` / ``smiles_tokenize``)
    to inspect each atom's original ``parsed_mapping`` token **before**
    parsing — bypassing the SMARTS parser's drop of that information and
    the per-side auto-counter that can otherwise hide fully-unmapped
    SMARTS rules behind coincidental atom-number collisions.

    Accepts both reaction SMILES (``reactants>>products`` or
    ``reactants>reagents>products``) and reaction SMARTS. Reagents,
    hydrogens, ring-closure digits and bonds are skipped — only heavy
    atoms count.

    Returns ``"fully_mapped"`` only when every heavy atom on both sides
    has an explicit map number *and* at least one map appears on both
    sides; ``"partially_mapped"`` when some atoms are bare; ``"unmapped"``
    when the explicit map sets do not overlap.

    :raises ValueError: if ``text`` is not a recognizable reaction string
        (no ``>``, or more than two ``>`` separators).
    """
    parts = text.split(">")
    if len(parts) == 2:
        reactants_text, products_text = parts
    elif len(parts) == 3:
        reactants_text, _reagents_text, products_text = parts
    else:
        raise ValueError(
            f"malformed reaction string {text!r}: expected one or two '>' "
            f"separators, got {len(parts) - 1}"
        )

    def _atom_maps(side: str) -> list[int | None]:
        # Atom tokens come back as type 0 (organic/bracketed) or 8 (aromatic
        # bare); other tokens (bonds, ring closures, dots, branch parens)
        # are filtered out. smarts_tokenize accepts SMILES too (strict
        # superset).
        side = _CXSMILES_BLOCK.sub("", side).strip()
        if not side:
            return []
        tokens = smarts_tokenize(side)
        return [
            payload.get("parsed_mapping")
            for token_type, payload in tokens
            if token_type in (0, 8) and isinstance(payload, dict)
        ]

    r_maps = _atom_maps(reactants_text)
    p_maps = _atom_maps(products_text)

    r_explicit = {m for m in r_maps if m}
    p_explicit = {m for m in p_maps if m}
    if not (r_explicit & p_explicit):
        return "unmapped"

    if any(not m for m in r_maps) or any(not m for m in p_maps):
        return "partially_mapped"
    return "fully_mapped"


def is_reaction_atom_mapped(reaction: ReactionContainer | str) -> bool:
    """Strict predicate — ``True`` only when every heavy atom is explicitly mapped.

    Accepts a parsed ``ReactionContainer`` or a raw reaction string
    (SMILES or SMARTS). Strings are routed through
    :func:`reaction_string_mapping_status` for reliable partial detection
    on SMARTS rules.
    """
    if isinstance(reaction, str):
        return reaction_string_mapping_status(reaction) == "fully_mapped"
    return reaction_mapping_status(reaction) == "fully_mapped"


def assert_reaction_atom_mapped(
    reaction: ReactionContainer | str, *, allow_partial: bool = True
) -> None:
    """Raise ``MappingError`` on fully unmapped reactions; warn on partial.

    :param reaction: parsed ``ReactionContainer`` (from ``chython.smiles``
        or ``chython.smarts``) **or** a raw reaction string (SMILES or
        SMARTS). Prefer the raw-string form for SMARTS rules: partial
        detection on a parsed SMARTS rule is unreliable because chython
        drops the ``parsed_mapping`` trace during SMARTS parsing.
    :param allow_partial: if ``True`` (default), emit a ``UserWarning``
        for partially-mapped reactions; if ``False``, raise
        ``MappingError``. Partial mapping is common from RDKit / RDChiral
        rule output and can produce wrong leaving/incoming group
        identification during rule extraction.
    :raises MappingError: when the reaction has no shared atom numbers
        between reactants and products, or when ``allow_partial=False``
        and the reaction is partially mapped.
    """
    if isinstance(reaction, str):
        status = reaction_string_mapping_status(reaction)
    else:
        status = reaction_mapping_status(reaction)
    if status == "unmapped":
        raise MappingError(
            "Reaction has no shared atom numbers between reactants and "
            "products; rule extraction and CGR composition will produce "
            "degenerate output."
        )
    if status == "partially_mapped":
        message = (
            "Reaction is only partially atom-mapped (some heavy atoms have "
            "no map number). Common from RDKit/RDChiral output; may produce "
            "wrong leaving/incoming groups during rule extraction."
        )
        if not allow_partial:
            raise MappingError(message)
        warnings.warn(message, stacklevel=2)


def validate_and_canonicalize(
    molecule: MoleculeContainer,
) -> MoleculeContainer | None:
    """Validate + canonicalize a CGR-rebuilt molecule in one kekule pass.

    Used by ``apply_reaction_rule`` on the ``rebuild_with_cgr=True``
    path, where CGR decompose bypasses ``CanonicalRetroReactor._patcher``.
    Drops on any error (matches ``_patcher``'s strict rejection).

    For user inputs (targets, building blocks), use the permissive
    ``safe_canonicalization`` instead.
    """
    # Atom-key sort, idempotent across calls.
    molecule._atoms = dict(sorted(molecule._atoms.items()))
    tmp = molecule.copy()
    try:
        tmp.remove_coordinate_bonds(keep_to_terminal=False)
        tmp.kekule()
        if tmp.check_valence():
            return None
        tmp.standardize(_fix_stereo=False)
        tmp.implicify_hydrogens(_fix_stereo=False)
        tmp.thiele(fix_tautomers=True)
        tmp.standardize_charges(prepare_molecule=False)
        tmp.standardize_tautomers(prepare_molecule=False)
        tmp.clean_stereo()
        return tmp
    except InvalidAromaticRing:
        return None


def hash_from_reaction_rule(reaction_rule: ReactionContainer) -> int:
    """Generates hash for the given reaction rule.

    :param reaction_rule: The reaction rule to be converted.
    :return: The resulting hash.
    """

    reactants_hash = tuple(sorted(hash(r) for r in reaction_rule.reactants))
    reagents_hash = tuple(sorted(hash(r) for r in reaction_rule.reagents))
    products_hash = tuple(sorted(hash(r) for r in reaction_rule.products))

    return hash((reactants_hash, reagents_hash, products_hash))


def reverse_reaction(
    reaction: ReactionContainer,
) -> ReactionContainer:
    """Reverses the given reaction.

    :param reaction: The reaction to be reversed.
    :return: The reversed reaction.
    """
    reversed_reaction = ReactionContainer(
        reaction.products, reaction.reactants, reaction.reagents, reaction.meta
    )
    reversed_reaction.name = reaction.name

    return reversed_reaction


# Re-exports of QueryCGR helpers from the representation package, done lazily
# via module ``__getattr__`` (PEP 562) to avoid an import-time cycle.
_REPRESENTATION_REEXPORTS = (
    "canonical_query_cgr_key",
    "cgr_from_reaction_rule",
    "compress_labels",
    "query_cgr_atom_label",
    "query_cgr_bond_label",
    "query_to_mol",
    "reaction_query_to_reaction",
)


def __getattr__(name: str):
    if name in _REPRESENTATION_REEXPORTS:
        from synplan.chem.reaction.rules.representation import query_cgr

        return getattr(query_cgr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
