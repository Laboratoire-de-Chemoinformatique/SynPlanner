"""Reaction classifier module for protection strategy analysis.

Classifies reactions into broad types based on CGR (Condensed Graph of Reaction)
bond analysis using chython. Used downstream by the route scanner to assess
functional group incompatibilities at each synthetic step.
"""

import logging
from typing import Optional, Set

from chython.containers import ReactionContainer
from chython.containers.cgr import CGRContainer

logger = logging.getLogger(__name__)


def get_reaction_center_atoms(
    reaction: ReactionContainer,
    cgr: Optional[CGRContainer] = None,
) -> Set[int]:
    """Extract atoms involved in bond or charge changes using CGR.

    Uses ``reaction.compose()`` to obtain the CGR, then returns atoms
    where bond orders or charges change.  ``cgr.center_atoms`` already
    includes both bond-change and charge-change atoms.

    :param reaction: A chython ReactionContainer representing a chemical reaction.
    :param cgr: Pre-composed CGR.  If ``None``, computed from *reaction*.
    :return: A set of atom indices that participate in bond or charge changes.
    """
    if cgr is None:
        cgr = ~reaction
    return set(cgr.center_atoms)


def _analyze_cgr_bonds(cgr):
    """Analyze CGR bonds and return formed, broken, changed counts and details.

    Returns a dict with:
      - formed, broken, changed: integer counts
      - formed_pairs: list of (atom1, atom2, bond) for newly formed bonds
      - broken_pairs: list of (atom1, atom2, bond) for broken bonds
      - changed_pairs: list of (atom1, atom2, bond) for changed bonds
    """
    formed = 0
    broken = 0
    changed = 0
    formed_pairs = []
    broken_pairs = []
    changed_pairs = []

    for atom1, atom2, bond in cgr.bonds():
        if bond.order is None and bond.p_order is not None:
            formed += 1
            formed_pairs.append((atom1, atom2, bond))
        elif bond.order is not None and bond.p_order is None:
            broken += 1
            broken_pairs.append((atom1, atom2, bond))
        elif bond.order != bond.p_order:
            changed += 1
            changed_pairs.append((atom1, atom2, bond))

    return {
        "formed": formed,
        "broken": broken,
        "changed": changed,
        "formed_pairs": formed_pairs,
        "broken_pairs": broken_pairs,
        "changed_pairs": changed_pairs,
    }


def classify_reaction_type_broad(
    reaction: ReactionContainer,
    cgr=None,
) -> str:
    """Classify a reaction into a broad type based on CGR bond analysis.

    This is the original 4-category classifier.

    Possible return values:
        - ``'bond_formation'``: only new bonds are formed
        - ``'bond_breaking'``: only existing bonds are broken
        - ``'substitution'``: bonds are both formed and broken
        - ``'other'``: no bond changes detected

    :param reaction: A chython ReactionContainer representing a chemical reaction.
    :param cgr: Pre-composed CGR.  If ``None``, computed from *reaction*.
    :return: A string label for the broad reaction type.
    """
    if cgr is None:
        cgr = ~reaction

    info = _analyze_cgr_bonds(cgr)
    formed = info["formed"]
    broken = info["broken"]
    changed = info["changed"]

    if formed == 0 and broken == 0 and changed == 0:
        return "other"

    if broken == 0 and changed == 0:
        return "bond_formation"

    if formed == 0 and changed == 0:
        return "bond_breaking"

    return "substitution"


def classify_reaction_type_detailed(
    reaction: ReactionContainer,
    cgr=None,
) -> str:
    """Classify a reaction into a fine-grained named type using CGR analysis.

    Examines which atoms and bond-order changes are involved at the
    reaction center to return a more specific label.

    Possible return values:
        - ``'acylation'`` -- C=O at center + new C-N or C-O bond
        - ``'alkylation'`` -- new C-N, C-O, or C-S bond without C=O at center
        - ``'reduction'`` -- net bond order decrease
        - ``'oxidation'`` -- net bond order increase
        - ``'cross_coupling'`` -- new C-C bond formed
        - ``'amide_formation'`` -- new C-N bond + C=O at center
        - ``'ester_formation'`` -- new C-O bond + C=O at center
        - ``'halogenation'`` -- new C-halogen bond
        - ``'dehalogenation'`` -- C-halogen bond broken
        - ``'ring_closure'`` -- intramolecular bond formation creating a ring
        - ``'ring_opening'`` -- ring bond broken
        - ``'other'`` -- fallback

    :param reaction: A chython ReactionContainer representing a chemical reaction.
    :param cgr: Pre-composed CGR.  If ``None``, computed from *reaction*.
    :return: A string label for the detailed reaction type.
    """
    if cgr is None:
        cgr = ~reaction

    info = _analyze_cgr_bonds(cgr)
    formed = info["formed"]
    broken = info["broken"]
    changed = info["changed"]

    if formed == 0 and broken == 0 and changed == 0:
        return "other"

    # Build an atom-symbol lookup from the CGR
    atom_symbols = {}
    for n, atom in cgr.atoms():
        atom_symbols[n] = atom.atomic_symbol

    halogens = {"F", "Cl", "Br", "I"}

    # Helper: check if a C=O bond exists at any center atom
    center_atoms = set(cgr.center_atoms)

    def _has_carbonyl_at_center():
        """Check if there is a C=O bond (unchanged) involving a center atom."""
        for a1, a2, bond in cgr.bonds():
            syms = {atom_symbols.get(a1, ""), atom_symbols.get(a2, "")}
            if syms == {"C", "O"} and bond.order == bond.p_order:
                if bond.order == 2:  # double bond C=O preserved
                    if a1 in center_atoms or a2 in center_atoms:
                        return True
        # Also check for C=O among changed bonds (order or p_order == 2)
        for a1, a2, bond in info["changed_pairs"]:
            syms = {atom_symbols.get(a1, ""), atom_symbols.get(a2, "")}
            if syms == {"C", "O"}:
                return True
        return False

    # Classify formed bonds by atom types
    formed_cc = []
    formed_cn = []
    formed_co = []
    formed_cs = []
    formed_c_hal = []
    for a1, a2, bond in info["formed_pairs"]:
        s1 = atom_symbols.get(a1, "")
        s2 = atom_symbols.get(a2, "")
        pair = {s1, s2}
        if pair == {"C"}:
            formed_cc.append((a1, a2, bond))
        elif pair == {"C", "N"}:
            formed_cn.append((a1, a2, bond))
        elif pair == {"C", "O"}:
            formed_co.append((a1, a2, bond))
        elif pair == {"C", "S"}:
            formed_cs.append((a1, a2, bond))
        elif "C" in pair and (pair - {"C"}) & halogens:
            formed_c_hal.append((a1, a2, bond))

    # Classify broken bonds by atom types
    broken_c_hal = []
    for a1, a2, bond in info["broken_pairs"]:
        s1 = atom_symbols.get(a1, "")
        s2 = atom_symbols.get(a2, "")
        pair = {s1, s2}
        if "C" in pair and (pair - {"C"}) & halogens:
            broken_c_hal.append((a1, a2, bond))

    # Check for ring closure / ring opening using product ring information
    # Ring closure: a formed bond where both atoms end up in the same ring
    # Ring opening: a broken bond that was part of a ring
    has_ring_closure = False
    has_ring_opening = False

    # For ring closure, check if formed bond creates a cycle in products
    for a1, a2, bond in info["formed_pairs"]:
        # Check if both atoms are in the same ring in the CGR
        # (product-side ring membership)
        if a1 in center_atoms and a2 in center_atoms:
            # Simple heuristic: if there is already a path between a1 and a2
            # via bonds that exist in products, this is a ring closure
            try:
                for ring in cgr.sssr:
                    if a1 in ring and a2 in ring:
                        has_ring_closure = True
                        break
            except Exception:
                pass
        if has_ring_closure:
            break

    for a1, a2, bond in info["broken_pairs"]:
        try:
            for ring in cgr.sssr:
                if a1 in ring and a2 in ring:
                    has_ring_opening = True
                    break
        except Exception:
            pass
        if has_ring_opening:
            break

    # Net bond order change for oxidation/reduction detection
    net_order_change = 0
    for a1, a2, bond in info["changed_pairs"]:
        old_order = bond.order if bond.order is not None else 0
        new_order = bond.p_order if bond.p_order is not None else 0
        net_order_change += new_order - old_order
    for a1, a2, bond in info["formed_pairs"]:
        new_order = bond.p_order if bond.p_order is not None else 0
        net_order_change += new_order
    for a1, a2, bond in info["broken_pairs"]:
        old_order = bond.order if bond.order is not None else 0
        net_order_change -= old_order

    has_carbonyl = _has_carbonyl_at_center()

    # --- Classification logic (most specific first) ---

    # Cross-coupling: new C-C bond (check before halogenation so that
    # Suzuki/Heck couplings that also break a C-halogen bond are not
    # misclassified as dehalogenation)
    if formed_cc:
        return "cross_coupling"

    # Halogenation / dehalogenation (only when no C-C bond formed)
    if formed_c_hal and not broken_c_hal:
        return "halogenation"
    if broken_c_hal and not formed_c_hal:
        return "dehalogenation"

    # Amide formation: new C-N bond + C=O at center
    if formed_cn and has_carbonyl:
        return "amide_formation"

    # Ester formation: new C-O bond + C=O at center
    if formed_co and has_carbonyl:
        return "ester_formation"

    # Ring closure / opening
    if has_ring_closure:
        return "ring_closure"
    if has_ring_opening:
        return "ring_opening"

    # Alkylation: new C-N, C-O, or C-S bond without carbonyl at center
    if formed_cn or formed_co or formed_cs:
        return "alkylation"

    # Oxidation / reduction based on net bond order change
    if net_order_change > 0:
        return "oxidation"
    if net_order_change < 0:
        return "reduction"

    return "other"


def classify_reaction_type(
    reaction: ReactionContainer,
    cgr=None,
) -> str:
    """Classify a reaction into a broad type based on CGR bond analysis.

    This is the default classifier used throughout the protection module.
    It delegates to :func:`classify_reaction_type_broad`.

    :param reaction: A chython ReactionContainer representing a chemical reaction.
    :param cgr: Pre-composed CGR.  If ``None``, computed from *reaction*.
    :return: A string label for the broad reaction type.
    """
    return classify_reaction_type_broad(reaction, cgr=cgr)
