"""Module containing additional functions needed in different reaction data processing
protocols."""

import logging
import re
import warnings
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
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


class StereoDiscardedWarning(UserWarning):
    """Stereochemistry present on the input was about to be discarded.

    Promote it to an error with
    ``warnings.simplefilter("error", StereoDiscardedWarning)`` to refuse
    stereo-bearing input instead of flattening it.
    """


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

    cleaned = clean_molecule(
        molecule,
        standardize=standardize,
        clean_stereo=clean_stereo,
        clean2d=clean2d,
    )
    if cleaned is molecule:
        logging.warning(
            "chython was not able to standardize molecule due to invalid aromatic ring"
        )
    return cleaned


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


def in_atom_order(molecule: MoleculeContainer) -> MoleculeContainer:
    """A copy whose atoms and bonds are stored in numeric order.

    chython spells a SMILES by walking those two dicts, so the same molecule
    reached two ways -- out of the reactor, or parsed back from a file -- spells
    itself the same way only once both are stored the same way. Without it a
    file rewritten from what it wrote does not match it, and the difference is
    a permutation of equivalent atoms that means nothing.
    """
    molecule = molecule.copy()
    molecule._atoms = dict(sorted(molecule._atoms.items()))
    molecule._bonds = {
        atom: dict(sorted(bonds.items()))
        for atom, bonds in sorted(molecule._bonds.items())
    }
    return molecule


def normalise(molecule: MoleculeContainer) -> MoleculeContainer | None:
    """The one form a route holds a molecule in, or ``None`` when chython cannot.

    Aromaticity is re-perceived rather than trusted, because a file may come from
    a tool whose model differs. Runs on a copy, so asking cannot rewrite the
    caller's molecule. Atom order is redone too: it does not change the SMILES of
    a molecule the reactor built, but it does for one rebuilt from a CGR, where
    the atoms went in in whatever order the composition left them.
    """
    molecule = in_atom_order(molecule)
    try:
        molecule.kekule()
        molecule.implicify_hydrogens()
        molecule.thiele()
    except InvalidAromaticRing:
        return None
    return molecule


def molecule_key(molecule: MoleculeContainer) -> str:
    """The one SMILES a route keys a molecule by, in stock and in the file alike.

    A molecule chython cannot prepare keeps its own spelling: route export is
    best-effort.
    """
    return str(normalise(molecule) or molecule)


def mapped_smiles(reaction: ReactionContainer) -> str:
    """The mapped SMILES a reaction node carries.

    Written from molecules in atom order, so the file a route writes is the file
    reading it back writes again -- otherwise every export permutes the atoms of
    some symmetric group and the two files diff for no chemical reason.
    """
    ordered = ReactionContainer(
        [in_atom_order(molecule) for molecule in reaction.reactants],
        [in_atom_order(molecule) for molecule in reaction.products],
        [in_atom_order(molecule) for molecule in reaction.reagents],
    )
    return format(ordered, "m")


def _warn_stereo_loss(molecule: MoleculeContainer) -> None:
    """Warn once per call site when ``clean_stereo`` is about to discard real stereo marks.

    chython only keeps a descriptor on a genuine stereocentre, so a surviving
    ``atom.stereo``/``bond.stereo`` is the definition of "real". The message is
    deliberately molecule-independent so the default warnings filter collapses a
    batch run to one line per call site.
    """
    if not (
        any(a.stereo is not None for _, a in molecule.atoms())
        or any(b.stereo is not None for *_, b in molecule.bonds())
    ):
        return
    warnings.warn(
        "Input stereochemistry is being discarded because this caller requested "
        "clean_stereo=True. Pass clean_stereo=False for stereo-aware planning or "
        "InChIKey catalogue preparation.",
        StereoDiscardedWarning,
        stacklevel=3,
    )


def clean_molecule(
    molecule: MoleculeContainer,
    *,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Clean a Chython molecule on a copy while preserving failure semantics.

    Returns the molecule it was *given* when chython cannot prepare an aromatic
    ring, so ``result is molecule`` is how a caller learns that nothing was
    cleaned. :func:`safe_canonicalization` is this function with ``clean2d``
    off -- measured identical on every molecule of a real search.
    """

    tmp = molecule.copy()
    try:
        tmp.remove_coordinate_bonds(keep_to_terminal=False)
        if standardize:
            tmp.canonicalize()
        if clean_stereo:
            _warn_stereo_loss(tmp)
            tmp.clean_stereo()
        if clean2d:
            tmp.clean2d()
        return tmp
    except InvalidAromaticRing:
        return molecule


def safe_canonicalization(
    molecule: MoleculeContainer, *, clean_stereo: bool = True
) -> MoleculeContainer:
    """The one spelling of a molecule: canonical, flat, without 2D coordinates.

    The building-block catalogue is written with this, so it is also what a
    lookup against the catalogue has to be written with.

    :param molecule: The given molecule to be canonicalized.
    :return: The canonicalized molecule, or the molecule itself when chython
        cannot prepare its aromatic ring.
    """
    molecule = molecule.copy()
    molecule._atoms = dict(sorted(molecule._atoms.items()))
    return clean_molecule(
        molecule,
        clean_stereo=clean_stereo,
        clean2d=False,
    )


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
    tmp = molecule.copy()
    tmp._atoms = dict(sorted(tmp._atoms.items()))
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
        tmp.fix_stereo()
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

    if Path(output_file).suffix.lower() == ".json":
        from synplan.chem.building_blocks import standardize_building_block_catalogue

        return standardize_building_block_catalogue(input_file, output_file)

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


def standardize_sdf_text(block: str) -> list[str]:
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


def standardize_smiles_batch(batch: list[str]) -> list[str]:
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
