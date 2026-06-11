"""Module containing additional functions needed in different reaction data processing
protocols."""

import logging
import re
import warnings
from collections.abc import Iterable
from io import StringIO
from typing import Literal

from chython import smiles as smiles_parser
from chython.containers import (
    MoleculeContainer,
    ReactionContainer,
)
from chython.exceptions import InvalidAromaticRing, MappingError
from chython.files.daylight.tokenize import smarts_tokenize
from chython.files.SDFrw import SDFRead
from tqdm.auto import tqdm

from synplan.utils.files import MoleculeReader, MoleculeWriter

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


def mol_from_smiles(
    smiles: str,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Converts a SMILES string to a `MoleculeContainer` object and optionally
    standardizes, cleans stereochemistry, and cleans 2D coordinates.

    :param smiles: The SMILES string representing the molecule.
    :param standardize: Whether to standardize the molecule (default is True).
    :param clean_stereo: Whether to remove the stereo marks on atoms of the molecule (default is True).
    :param clean2d: Whether to clean the 2D coordinates of the molecule (default is True).
    :return: The processed molecule object.
    :raises ValueError: If the SMILES string could not be processed by chython.
    """
    molecule = smiles_parser(smiles, ignore=True)

    if not isinstance(molecule, MoleculeContainer):
        raise ValueError("SMILES string was not processed by chython")

    tmp = molecule.copy()
    try:
        tmp.remove_coordinate_bonds(keep_to_terminal=False)
        if standardize:
            tmp.canonicalize()
        if clean_stereo:
            tmp.clean_stereo()
        if clean2d:
            tmp.clean2d()
        molecule = tmp
    except InvalidAromaticRing:
        logging.warning(
            "chython was not able to standardize molecule due to invalid aromatic ring"
        )
    return molecule


def unite_molecules(molecules: Iterable[MoleculeContainer]) -> MoleculeContainer:
    """Unites a list of MoleculeContainer objects into a single MoleculeContainer. This
    function takes multiple molecules and combines them into one larger molecule. The
    first molecule in the list is taken as the base, and subsequent molecules are united
    with it sequentially.

    :param molecules: A list of MoleculeContainer objects to be united.
    :return: A single MoleculeContainer object representing the union of all input
        molecules.
    """
    new_mol = MoleculeContainer()
    for mol in molecules:
        new_mol = new_mol.union(mol)
    return new_mol


def safe_canonicalization(molecule: MoleculeContainer) -> MoleculeContainer:
    """Attempts to canonicalize a molecule, handling any exceptions. If the
    canonicalization process fails due to an InvalidAromaticRing exception, it safely
    returns the original molecule.

    :param molecule: The given molecule to be canonicalized.
    :return: The canonicalized molecule if successful, otherwise the original molecule.
    """
    molecule._atoms = dict(sorted(molecule._atoms.items()))

    molecule_copy = molecule.copy()
    try:
        molecule_copy.remove_coordinate_bonds(keep_to_terminal=False)
        molecule_copy.canonicalize()
        molecule_copy.clean_stereo()
        return molecule_copy
    except InvalidAromaticRing:
        return molecule


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


def standardize_building_blocks(input_file: str, output_file: str) -> str:
    """Standardizes custom building blocks.

    :param input_file: The path to the file that stores the original building blocks.
    :param output_file: The path to the file that will store the standardized building
        blocks.
    :return: The path to the file with standardized building blocks.
    """
    if input_file == output_file:
        raise ValueError("input_file name and output_file name cannot be the same.")

    with (
        MoleculeReader(input_file) as inp_file,
        MoleculeWriter(output_file) as out_file,
    ):
        for mol in tqdm(
            inp_file,
            desc="Number of building blocks processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            try:
                mol = safe_canonicalization(mol)
            except Exception as e:
                logging.debug(e)
                continue
            out_file.write(mol)

    return output_file


def _standardize_one_smiles(smiles_str: str) -> str | None:
    try:
        mol = smiles_parser(smiles_str, ignore=True)
        mol = safe_canonicalization(mol)
        return str(mol)
    except Exception:
        return None


def _standardize_sdf_range(filename: str, start: int, end: int) -> list[str]:
    out: list[str] = []
    sdf = SDFRead(filename, indexable=True)
    try:
        for i in range(start, end):
            try:
                mol = sdf[i]
                mol = safe_canonicalization(mol)
                out.append(str(mol))
            except Exception:
                pass
    finally:
        sdf.close()
    return out


def _standardize_sdf_text(block: str) -> list[str]:
    """Standardize molecules from an SDF text block.

    The block may contain one or multiple SDF records, separated by $$$$ lines.
    """
    out: list[str] = []
    with StringIO(block) as fh, SDFRead(fh) as sdf:
        for mol in sdf:
            try:
                mol = safe_canonicalization(mol)
                out.append(str(mol))
            except Exception:
                # ignore malformed entries
                pass
    return out


def _standardize_smiles_batch(batch: list[str]) -> list[str]:
    """Standardize a batch of SMILES strings and return valid results."""
    out: list[str] = []
    for smiles_str in batch:
        res = _standardize_one_smiles(smiles_str)
        if res:
            out.append(res)
    return out


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
