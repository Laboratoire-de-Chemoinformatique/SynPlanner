"""Imputation of missing molecules in unbalanced reactions.

Inspired by the SynRBL strategy (10.1186/s13321-024-00875-4) on chython: a
rule-based pass matches a non-carbon element deficit against a table of small
molecules, and a substructure pass recovers missing carbon fragments. The
substructure search matches atoms by Morgan environment hash counts instead of
a maximum common subgraph, which is approximate but linear in the atom count.
"""

from collections import Counter, defaultdict
from contextlib import suppress
from functools import cache, lru_cache
from itertools import combinations

from chython import smarts as smarts_chython
from chython import smiles as smiles_chython
from chython.containers import MoleculeContainer, ReactionContainer
from chython.exceptions import InvalidAromaticRing

MAX_RADIUS = 4
_MAX_CANDIDATES = 60
_MAX_SEARCH_NODES = 5_000
_MAX_MERGE_TRIALS = 64


def _formula(molecules) -> Counter:
    """Count elements (hydrogens included) and net charge under ``"Q"``."""
    formula: Counter = Counter()
    for molecule in molecules:
        formula.update(molecule.brutto)
        formula["Q"] += sum(atom.charge for _, atom in molecule.atoms())
    return formula


def _table(*smiles_strings: str) -> tuple[tuple[str, dict[str, int]], ...]:
    """Pair each SMILES with the composition chython reads off it.

    Writing the counts out alongside would be a second spelling of the same
    fact, free to drift from the first.  Richer compositions sort first so a
    deficit is covered by one specific species rather than several generic ones.
    """
    entries = (
        (text, dict(_formula([smiles_chython(text)]))) for text in smiles_strings
    )
    return tuple(sorted(entries, key=lambda entry: -len(entry[1])))


# Species the rule-based pass may add to either side, taken from SynRBL's rule
# table.  Compositions include hydrogens and a net charge under "Q", matching
# the keys of :func:`_formula`.  Richer compositions are tried first, so a
# deficit is covered by one specific species rather than several generic ones.
SMALL_MOLECULES = _table(
    "NS(=O)(=O)Cl",
    "NS(N)(=O)=O",
    "O=S(=O)(O)Cl",
    "OS(Cl)=O",
    "B(O)(O)Cl",
    "B(O)(O)Br",
    "B(O)(O)I",
    "S(Br)(O)=O",
    "O=S(Cl)Cl",
    "B(O)(O)O",
    "B(O)(O)",
    "[NH3+]O",
    "NO",
    "C=O",
    "[OH-]",
    "[N+](=O)([O-])[O-]",
    "[O-]I(=O)=O",
    "O=N[O-]",
    "[O-]S(=O)(=O)[O-]",
    "[O-]P(=O)([O-])[O-]",
    "[O-]S(=O)[O-]",
    "[NH2-]",
    "[NH4+]",
    "O",
    "N",
    "O=S=O",
    "F[P-](F)(F)(F)(F)F",
    "F[B-](F)(F)F",
    "[H][H]",
    "[H]",
    "N#N",
    "[N-]=[N+]=[N-]",
    "[H+]",
    "[F-]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[S-2]",
    "[Na+]",
    "[Li+]",
    "[K+]",
    "[Ca+2]",
    "[Mg+2]",
    "[Ba+2]",
    "[Al+3]",
    "[Zn+2]",
    "[Cu+2]",
    "[Cu+]",
    "[Cs+]",
)

# Reachable only once a deficit is being completed with hydrogen: a halogen or
# sulfur that took one up is written as the acid, not as an ion beside a loose
# hydrogen atom.  Offering these to an exact match would hide the salts.
PROTONATED = _table(
    "F",
    "Cl",
    "Br",
    "I",
    "S",
)

# Reachable only once a deficit is being completed with hydroxide.  A metal
# does not leave as a bare "MgBr" — magnesium is divalent, so that fragment has
# an unsatisfied valence, which is what a CGR round trip produces here and what
# `check_valence` rejects.  It leaves the aqueous workup as the hydroxide.
LEAVING_GROUPS = _table(
    "O[Mg]Cl",
    "O[Mg]Br",
    "O[Mg]I",
    "O[Zn]Cl",
    "O[Zn]Br",
    "O[Mg]O",
    "O[Mg+]",
    "[Li]O",
    "[Na]O",
    "[K]O",
    "Cl[Mg]Cl",
    "Br[Mg]Br",
    "I[Mg]I",
    "Cl[Zn]Cl",
    "Br[Zn]Br",
)

# Pyridinium chlorochromate and what it becomes, the conventional stand-in for
# an oxidation recorded without its oxidant.  Per unit the product side gains
# two hydrogens over the reactant side, which is exactly the deficit of an
# alcohol written straight to the carbonyl.  Named only for that reaction:
# SynRBL's rule table reaches for it on an alcohol or aldehyde and leaves every
# other oxidation as the bare oxygen atom.
OXIDANT_COUPLE: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("O", "O=[Cr](=O)([O-])Cl", "c1cccc[nH+]1"),
    ("O", "O[Cr](O)=O", "[Cl-]", "c1cccc[nH+]1"),
)

# What an anion and a proton make when they leave together rather than as a
# dissociated pair: the acid, or in hydroxide's case water.
PROTON_PARTNERS: dict[str, str] = {
    "[F-]": "F",
    "[Cl-]": "Cl",
    "[Br-]": "Br",
    "[I-]": "I",
    "[OH-]": "O",
}

# An oxidant recorded as nothing but its oxygen.  Needs `keep_implicit` to
# parse, so it is named rather than spelled out at each use.
ATOMIC_OXYGEN = "[O]"

# The functional groups expand rules ask about, as
# ``(patterns, anti-patterns)``.
# A group holds when a pattern matches at the anchor and no anti-pattern does.
FUNCTIONAL_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # chython's own cleavage pattern (chython/reactor/scaffold.py): a degree-one
    # sp3 oxygen on an sp3 carbon, which excludes phenol, acid and ether without
    # a list of anti-patterns.
    "alcohol": (("[O;D1;z1;y0][C;z1;y1]",), ()),
    "ether": (("COC",), ("OC=S", "C=O", "C=CO", "OCOC", "OC=O", "OCN")),
    "thioether": (("CSC",), ("O=CS",)),
}

# What a named reagent becomes once it has done its job.  These by-products are
# only offered when the reagent itself is in the record, so they cannot be
# reached for a reaction that never used one: the reagent is the evidence.  A
# patent names what it weighed out, and its spent form is stoichiometry rather
# than a guess, which is what separates these from the substructure search.
REAGENT_BYPRODUCTS: dict[str, str] = {
    "CC(=O)OC(C)=O": "CC(=O)O",  # acetic anhydride -> acetic acid
    "O=C(OC(=O)C(F)(F)F)C(F)(F)F": "OC(=O)C(F)(F)F",  # TFAA -> TFA
    "O=C1CCC(=O)N1Br": "O=C1CCC(=O)N1",  # NBS -> succinimide
    "O=C1CCC(=O)N1Cl": "O=C1CCC(=O)N1",  # NCS -> succinimide
    "O=C(OO)c1cccc(Cl)c1": "O=C(O)c1cccc(Cl)c1",  # mCPBA -> m-chlorobenzoic acid
    # Sulfonylating an alcohol only breaks its O-H and expels chloride, which
    # the halide path already covers.  These are for the other case, where the
    # sulfonyl was a transient activator and left as the whole sulfonate.
    "CS(=O)(=O)Cl": "CS(=O)(=O)O",  # MsCl -> methanesulfonic acid
    "Cc1ccc(S(=O)(=O)Cl)cc1": "Cc1ccc(S(=O)(=O)O)cc1",  # TsCl -> tosic acid
}


# Species the formula match can reach that do not survive in a flask, and what
# they fall apart into.  Each pair is element-neutral, so replacing one with
# the other leaves the balance untouched and only the chemistry improves.
UNSTABLE: dict[str, tuple[str, ...]] = {
    "OC(O)=O": ("O=C=O", "O"),  # carbonic acid: the CO2 vents
    "OS(Cl)=O": ("O=S=O", "Cl"),  # chlorosulfinic acid: SO2 and HCl vent
}


# Elements whose presence among the recorded reagents makes a redox step
# credible: a transition metal catalyst, or the boron and aluminium of a
# hydride reagent.  A patent that hydrogenated something names what it used.
# The alkali metals are deliberately absent: a bare counter-ion says nothing,
# and the boron and aluminium of a borohydride or alanate are already here.
_REDOX_ELEMENTS = frozenset(
    (
        "Pd",
        "Pt",
        "Ni",
        "Rh",
        "Ru",
        "Ir",
        "Fe",
        "Zn",
        "Sn",
        "Mn",
        "Cr",
        "Cu",
        "Co",
        "Ti",
        "B",
        "Al",
    )
)


# Elements that leave as the hydroxide rather than the hydride.  Phosphorus is
# among them: a phosphite ester that loses its carbon ends the workup as the
# phosphate, and a phosphonium ylide as the phosphine oxide.
_HYDROXIDE_FORMERS = frozenset({"Mg", "Zn", "Si", "B", "P"})

# Boundary pairs that must not be bonded to each other.  Two heteroatoms
# meeting at a cut almost always means the substructure match was wrong.
_NO_BOND = frozenset({"N", "O", "F", "Cl", "Br", "I"})

_HALOGENS = frozenset({"F", "Cl", "Br", "I"})


class RebalancingError(ValueError):
    """Raised when a reaction cannot be balanced."""


# How often an answer proves right, by how many molecules it had to invent,
# measured on SynRBL's validation set.  Inventing one or two is usually a real
# leaving group; inventing many means the search was guessing.
_CONFIDENCE_BY_ADDED = (0.95, 0.94, 0.88, 0.57, 0.50, 0.30, 0.25, 0.20, 0.10)


def confidence(original: ReactionContainer, balanced: ReactionContainer) -> float:
    """How far an answer sits inside the region where it is usually right.

    Not a probability from a fitted model — a reading of the few things that
    separate a reliable answer from a guess: how much had to be invented, how
    crowded the reaction is, and whether loose hydrogen was needed.

    :param original: The reaction as recorded.
    :param balanced: What :func:`rebalance_reaction` made of it.
    :return: A value in ``[0, 1]``; 0 means do not trust this.
    """
    if reaction_imbalance(original).get("C", 0) < 0:
        return 0.0  # solved backwards, and never yet seen to be right
    before = Counter(str(molecule) for molecule in original.molecules())
    after = Counter(str(molecule) for molecule in balanced.molecules())
    added = [smiles_chython(text) for text in (after - before).elements()]

    score = _CONFIDENCE_BY_ADDED[min(len(added), len(_CONFIDENCE_BY_ADDED) - 1)]
    if sum(1 for molecule in added if len(molecule) > 3) > 2:
        score *= 0.5  # several whole organic species conjured at once
    if len(list(original.molecules())) > 4:
        score *= 0.5  # crowded reactions misalign, as the paper also reports
    if any(set(molecule.brutto) == {"H"} for molecule in added):
        score *= 0.75  # loose hydrogen stands in for chemistry left unread
    return round(score, 3)


def _has_broken_valence(molecule: MoleculeContainer) -> bool:
    """Whether an atom is left holding fewer bonds than its valence requires.

    ``check_valence`` only means anything on a kekulized molecule — read
    straight from ``c1ccncc1`` pyridine reports a false error — so the molecule
    is kekulized first, on a copy.  A ring that will not kekulize at all is
    itself a broken structure.  Only the verdict is wanted, so none of the
    canonicalisation that follows a valence check elsewhere is done here.
    """
    kekulized = molecule.copy()
    kekulized._atoms = dict(sorted(kekulized._atoms.items()))
    try:
        kekulized.remove_coordinate_bonds(keep_to_terminal=False)
        kekulized.kekule()
        return bool(kekulized.check_valence())
    except InvalidAromaticRing:
        return True


def reaction_imbalance(reaction: ReactionContainer) -> dict[str, int]:
    """Element counts missing from the product side, negative where the
    reactant side is short.

    :param reaction: Input reaction.
    :return: Non-zero element counts of reactants minus products.
    """
    formula = _formula(reaction.reactants)
    formula.subtract(_formula(reaction.products))
    return {element: count for element, count in formula.items() if count}


def _atom_pool(molecules) -> list[dict[int, list[tuple[int, int]]]]:
    """Index every atom of ``molecules`` by its Morgan hash at each radius."""
    pool: list[dict[int, list[tuple[int, int]]]] = [
        defaultdict(list) for _ in range(MAX_RADIUS)
    ]
    for index, molecule in enumerate(molecules):
        for radius, layer in enumerate(molecule._morgan_hash_dict(1, MAX_RADIUS)):
            for atom, atom_hash in layer.items():
                pool[radius][atom_hash].append((index, atom))
    return pool


def _unmatched_atoms(reaction: ReactionContainer) -> list[set[int]]:
    """Reactant atoms with no counterpart among the products, per reactant.

    Atoms are paired with product atoms greedily from the widest environment
    down, across all reactants at once so that the pairing does not depend on
    the order the reactants happen to be listed in.  This approximates a
    maximum common substructure without the subgraph search.
    """
    pool = _atom_pool(reaction.products)
    claimed: set = set()
    layers = [
        molecule._morgan_hash_dict(1, MAX_RADIUS) for molecule in reaction.reactants
    ]
    matched: list[set[int]] = [set() for _ in reaction.reactants]
    for radius in range(MAX_RADIUS - 1, -1, -1):
        candidates = pool[radius]
        for index, molecule_layers in enumerate(layers):
            for atom, atom_hash in molecule_layers[radius].items():
                if atom in matched[index]:
                    continue
                free = candidates.get(atom_hash)
                while free:
                    slot = free.pop()
                    if slot in claimed:
                        continue
                    claimed.add(slot)
                    matched[index].add(atom)
                    break
    return [
        set(molecule) - found
        for molecule, found in zip(reaction.reactants, matched, strict=True)
    ]


def _broken_bonds(reaction: ReactionContainer) -> set[frozenset] | None:
    """Bonds the reactants lose, read off the CGR.

    Only meaningful for a mapped reaction: there the CGR says exactly which
    bonds break, which beats guessing across every bond in the molecule.
    Returns ``None`` when the reaction carries no usable mapping.

    :param reaction: Input reaction.
    :return: Atom pairs whose bond disappears, or ``None`` if unmapped.
    """
    numbers = [atom for molecule in reaction.reactants for atom in molecule]
    if len(numbers) != len(set(numbers)):
        return None  # per-molecule numbering, i.e. no mapping
    product_numbers = {atom for molecule in reaction.products for atom in molecule}
    if not product_numbers & set(numbers):
        return None
    try:
        cgr = ~ReactionContainer(reaction.reactants, reaction.products)
    except Exception:  # an unmappable reaction is simply not a mapped one
        return None
    return {
        frozenset((left, right))
        for left, right, bond in cgr.bonds()
        if bond.p_order is None
    }


def _side_of_cuts(molecule: MoleculeContainer, start: int, cuts) -> set[int]:
    """Atoms reachable from ``start`` once every bond in ``cuts`` is ignored."""
    bonds = molecule._bonds
    seen = {start}
    queue = [start]
    while queue:
        atom = queue.pop()
        for neighbour in bonds[atom]:
            if neighbour in seen or frozenset((atom, neighbour)) in cuts:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return seen


def _cut_candidates(molecule: MoleculeContainer, broken: set[frozenset] | None = None):
    """Every fragment the molecule falls into when one bond is broken.

    Ring bonds are skipped — breaking one leaves the molecule in one piece.
    The molecule as a whole is a candidate too, for reagents consumed entirely
    or carried through unchanged.  When the CGR has named the bonds that break,
    only those are tried: a reagent the reaction never touched then has no
    candidate but itself, instead of being cut apart and capped.

    :return: Iterator of ``(atoms, boundary, neighbour)`` where ``boundary`` is
        ``(atom, element that closes the broken bond)`` or ``None``, and
        ``neighbour`` is the atom left behind on the other side of the break.
    """
    yield set(molecule), None, None
    for left, right, bond in molecule.bonds():
        if bond.in_ring:
            continue
        if broken is not None and frozenset((left, right)) not in broken:
            continue
        piece = _side_of_cuts(molecule, left, {frozenset((left, right))})
        yield piece, (left, _cap_atom(molecule, left, right)), right
        yield set(molecule) - piece, (right, _cap_atom(molecule, right, left)), left
    yield from _double_cut_candidates(molecule, broken)


def _double_cut_candidates(molecule: MoleculeContainer, broken: set[frozenset] | None):
    """Fragments freed only when two bonds break together.

    A ring opens, or a protecting group comes off at both its anchors — neither
    shows up in the single-bond enumeration.  A carbon that loses two oxygens
    at once is an acetal or ketal, and departs as the ketone rather than as the
    diol two separate caps would give.
    """
    if not broken:
        return
    atoms = set(molecule)
    inside = [pair for pair in broken if pair <= atoms]
    for first, second in combinations(inside, 2):
        for anchor in first:
            piece = _side_of_cuts(molecule, anchor, {first, second})
            if not 0 < len(piece) < len(molecule):
                continue
            other = next(iter(first - {anchor}), anchor)
            if other in piece:
                continue
            partners = [next(iter(pair - {anchor})) for pair in (first, second)]
            if (
                anchor in second
                and molecule.atom(anchor).atomic_symbol == "C"
                and all(molecule.atom(p).atomic_symbol in ("O", "S") for p in partners)
            ):
                yield piece, (anchor, ("O", 2)), other
            else:
                yield piece, (anchor, _cap_atom(molecule, anchor, other)), other


@lru_cache(maxsize=128)
def _query(pattern: str):
    return smarts_chython(pattern)


def _matches_at(molecule: MoleculeContainer, pattern: str, atom: int) -> bool:
    """Whether ``pattern`` matches ``molecule`` covering ``atom``."""
    try:
        mappings = _query(pattern).get_mapping(molecule)
    except Exception:  # a pattern chython will not read is simply no match
        return False
    return any(atom in mapping.values() for mapping in mappings)


def _aromatic(molecule: MoleculeContainer) -> MoleculeContainer:
    """The molecule with its rings written as aromatic, numbering kept.

    Not cached: two molecules can be structurally equal and numbered
    differently, and the caller asks about a numbered atom.
    """
    aromatic = molecule.copy()
    with suppress(Exception):
        aromatic.thiele()
    return aromatic


def is_functional_group(molecule: MoleculeContainer, name: str, atom: int) -> bool:
    """Whether ``atom`` sits in the named functional group.

    Asked of the aromatic form: a reaction SMILES arrives Kekule, where
    ``C=CO`` — written to spot an enol ether — matches every aryl ether, and
    ``Oc1ccccc1`` — written to spot a phenol — matches none.

    :param molecule: Molecule the atom belongs to.
    :param name: Key of :data:`FUNCTIONAL_GROUPS`.
    :param atom: Atom the group must cover.
    """
    patterns, anti_patterns = FUNCTIONAL_GROUPS[name]
    molecule = _aromatic(molecule)
    if not any(_matches_at(molecule, p, atom) for p in patterns):
        return False
    return not any(_matches_at(molecule, p, atom) for p in anti_patterns)


def _cap_atom(
    molecule: MoleculeContainer, atom: int, neighbour: int
) -> tuple[str, int] | None:
    """What closes the bond broken between ``atom`` and ``neighbour``.

    Reproduces expand rules by looking at the graph directly: an ether
    or thioether cut leaves the alkyl halide, a bond to a heteroatom or to
    another carbon picks up oxygen, and everything else keeps the hydrogen
    chython fills in on its own.

    :return: ``(element, bond order)`` to attach, or ``None`` for a hydrogen.
    """
    symbol = molecule.atom(atom).atomic_symbol
    other = molecule.atom(neighbour).atomic_symbol
    # Expand rules, in their order: a dealkylation leaves the halide,
    # an acyl cleavage takes up oxygen, a metal takes hydroxide.
    if (
        symbol == "C"
        and other == "O"
        and is_functional_group(molecule, "ether", neighbour)
    ):
        return ("I", 1)
    if (
        symbol == "C"
        and other == "S"
        and is_functional_group(molecule, "thioether", neighbour)
    ):
        return ("I", 1)
    if symbol in _HYDROXIDE_FORMERS:
        return ("O", 1)
    if symbol != "C":
        return None
    # Silicon counts among them: a phenyl cut off a silane leaves as the
    # phenol, taking the oxygen the substrate gave up, not as benzene.
    return ("O", 1) if other in ("O", "S", "N", "C", "Si") else None


def _missing_molecules(
    reaction: ReactionContainer,
    deficit: dict[str, int],
    broken: set[frozenset] | None,
) -> MoleculeContainer:
    """The molecules the products are short of, rebuilt from the reactants.

    Which bond broke is decided by the element balance and the fingerprint
    match together: the balance fixes how much carbon has to leave, and among
    the fragments of that size the one whose atoms are least accounted for in
    the products wins.  Breaks that will not close into a valid molecule are
    passed over rather than capped, which is how a leaving group larger than
    the first plausible fragment gets found.

    :param broken: Bonds the mapping says break, or ``None`` where there is
        no mapping to read them off.
    :return: The molecules to add to the product side, as one container.
    :raises RebalancingError: If no break yields molecules that stand up.
    """
    unmatched = _unmatched_atoms(reaction)
    partner = _spare_water(reaction, unmatched)
    # The CGR's own breaks first, every bond only if none of them fits: a
    # mapped reaction says where it broke, but it cannot say so about the
    # reagents it never recorded.
    carbon_needed = deficit["C"]
    for cuts in [broken, None] if broken else [None]:
        merged = _fragments_for(reaction, unmatched, partner, deficit, cuts)
        if merged is not None:
            # An imputed fragment can come out as the enol of what it should
            # be; canonicalize settles the tautomer.  Stereo is kept, so this
            # is not `validate_and_canonicalize`, which strips it.
            with suppress(Exception):  # a fragment chython will not tidy stands
                merged.canonicalize()
            return merged
    raise RebalancingError(f"no fragment holds the {carbon_needed} missing carbons")


def _fragments_for(
    reaction: ReactionContainer,
    unmatched: list[set[int]],
    partner: tuple | None,
    deficit: dict[str, int],
    broken: set[frozenset] | None,
) -> MoleculeContainer | None:
    """Best combination of fragments cut at ``broken``, or None if none fits."""
    carbon_needed = deficit["C"]
    candidates = []
    for molecule, missing in zip(reaction.reactants, unmatched, strict=True):
        for atoms, boundary, neighbour in _cut_candidates(molecule, broken):
            carbons = sum(molecule.atom(atom).atomic_symbol == "C" for atom in atoms)
            if not 0 < carbons <= carbon_needed:
                continue
            # A break is only in the right place when the atom left behind is
            # itself accounted for in the products.  Cutting where both sides
            # are missing truncates a larger leaving group, and chython fills
            # the dangling carbon with a hydrogen rather than complaining.
            anchored = neighbour is None or neighbour not in missing
            candidates.append(
                (
                    anchored,
                    len(atoms & missing) / len(atoms),
                    carbons,
                    molecule,
                    atoms,
                    boundary,
                )
            )
    candidates.sort(key=lambda candidate: (not candidate[0], -candidate[1]))

    best, best_score, tried = None, None, 0
    for size in (1, 2, 3):
        for combination in combinations(candidates[:_MAX_CANDIDATES], size):
            if sum(item[2] for item in combination) != carbon_needed:
                continue
            # Two breaks may share a molecule so long as they take different
            # atoms of it: a diester loses both its acyl groups at once.
            if any(
                left[3] is right[3] and left[4] & right[4]
                for left, right in combinations(combination, 2)
            ):
                continue
            base = [
                (molecule.substructure(atoms), boundary)
                for _, _, _, molecule, atoms, boundary in combination
            ]
            for variant in _cap_variants(base, deficit):
                for use_partner in (True, False) if partner else (False,):
                    for merge in (_merge, _merge_apart):
                        fragments = [(piece.copy(), bound) for piece, bound in variant]
                        if (
                            use_partner
                            and partner is not None
                            and sum(b is not None for _, b in fragments) == 1
                        ):
                            fragments.append((partner[0].copy(), partner[1]))
                        try:
                            merged = merge(fragments)
                        except RebalancingError:
                            continue
                        pieces = merged.split()
                        if any(_has_broken_valence(piece) for piece in pieces):
                            continue  # the break truncated a larger leaving group
                        # Of the breaks that work, prefer the one accounting
                        # for most of what is missing: a leaving group the
                        # balance still has to top up with extras is the wrong
                        # guess.  Fewest breaks settles a tie: one bond
                        # breaking explains a reaction more plausibly than two
                        # happening to fit.
                        score = (_unexplained(merged, deficit), len(base), len(pieces))
                        if best is None or score < best_score:
                            best, best_score = merged, score
                        tried += 1
                        if score == (0, 1, 1) or tried >= _MAX_MERGE_TRIALS:
                            return best
        if best is not None:
            return best
    return best


def _cap_variants(base, deficit: dict[str, int]):
    """The fragments as cut, then again closed with what the balance still wants.

    A cross-coupling does not give off its halide as the acid: the boron, tin or
    silicon left holding an open bond takes the halogen the products are short
    of.  The caps :func:`_cap_atom` chose are offered first, so a deficit only
    ever settles a case those rules left worse explained.
    """
    yield base
    caps = [
        (element, 1) for element in ("Cl", "Br", "I") if deficit.get(element, 0) > 0
    ]
    if deficit.get("O", 0) > 0:
        caps.append(("O", 2))
    for cap in caps:
        if all(
            not boundary or _may_cap(piece, boundary[0], cap)
            for piece, boundary in base
        ):
            yield [
                (piece, (boundary[0], cap) if boundary else boundary)
                for piece, boundary in base
            ]


def _may_cap(fragment: MoleculeContainer, atom: int, cap: tuple[str, int]) -> bool:
    """Whether the open bond at ``atom`` may be closed with ``cap``.

    Only phosphorus takes a second oxygen — anywhere else that is a peroxide.
    A halogen answers to the same rule as a merge: put on a heteroatom it makes
    a hypohalite, which is never what a reaction quietly gave off.  Nor may one
    go on a carbon already holding two oxygens: an acyl halide is real, a
    carbonate halide is not, and carbonate is among the commonest reagents
    there is.
    """
    symbol = fragment.atom(atom).atomic_symbol
    if cap == ("O", 2):
        return symbol == "P"
    if cap[0] in _HALOGENS and symbol == "C":
        oxygens = sum(
            fragment.atom(neighbour).atomic_symbol == "O"
            for neighbour in fragment.int_adjacency[atom]
        )
        if oxygens > 1:
            return False
    return symbol not in _NO_BOND


def _merge_apart(fragments) -> MoleculeContainer:
    """Cap each open bond on its own rather than bonding the fragments.

    Two acyl groups leaving the same diester become two acids, not one
    molecule joining them end to end.
    """
    pieces = []
    for fragment, boundary in fragments:
        piece = fragment.copy()
        if boundary and boundary[1] is not None:
            atom, (element, order) = boundary
            piece.add_bond(atom, piece.add_atom(element), order)
        pieces.append(piece)
    if not pieces:
        raise RebalancingError("nothing to impute")
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.union(piece, remap=True)
    return merged


def _unexplained(merged: MoleculeContainer, deficit: dict[str, int]) -> int:
    """How many atoms of the deficit the imputed molecules fail to account for."""
    left = dict(deficit)
    for symbol, count in merged.brutto.items():
        left[symbol] = left.get(symbol, 0) - count
    return sum(abs(count) for symbol, count in left.items() if symbol != "Q")


def _spare_water(reaction: ReactionContainer, unmatched) -> tuple | None:
    """A reactant water with no counterpart in the products.

    Such a water is the reagent that completes a leaving group rather than a
    spectator, so it is handed back with an open bond on its oxygen.
    """
    for molecule, missing in zip(reaction.reactants, unmatched, strict=True):
        if molecule.brutto == {"O": 1, "H": 2} and missing:
            return molecule.copy(), (next(iter(molecule)), None)
    return None


def _join(
    left: MoleculeContainer,
    right: MoleculeContainer,
    left_atom,
    right_atom,
):
    """Union two fragments and bond the given atoms, tracking the renumbering."""
    renumbered = right.copy()
    mapping = {atom: i for i, atom in enumerate(right, start=max(left) + 1)}
    renumbered.remap(mapping)
    merged = left.union(renumbered)
    merged.add_bond(left_atom, mapping[right_atom], 1)
    return merged


def _alcohol_oxygen(molecule: MoleculeContainer) -> int | None:
    """The hydroxyl oxygen of an alcohol, if the molecule is one."""
    for atom, properties in molecule.atoms():
        if properties.atomic_symbol == "O" and is_functional_group(
            molecule, "alcohol", atom
        ):
            return atom
    return None


def _ketones(molecules) -> int:
    """Aldehyde and ketone carbonyls; an acid, ester or amide does not count."""
    total = 0
    for molecule in molecules:
        for left, right, bond in molecule.bonds():
            if bond.order != 2:
                continue
            symbols = {
                molecule.atom(left).atomic_symbol,
                molecule.atom(right).atomic_symbol,
            }
            if symbols != {"C", "O"}:
                continue
            carbon = left if molecule.atom(left).atomic_symbol == "C" else right
            if all(
                molecule.atom(neighbour).atomic_symbol == "C"
                for neighbour in molecule._bonds[carbon]
                if neighbour not in (left, right)
            ):
                total += 1
    return total


def _is_alcohol_oxidation(reaction: ReactionContainer) -> bool:
    """Whether a carbinol became a carbonyl, the one oxidation with a named
    oxidant.  Everything else that gives off hydrogen is left to
    :data:`ATOMIC_OXYGEN`.
    """
    if _ketones(reaction.products) <= _ketones(reaction.reactants):
        return False
    return any(_alcohol_oxygen(molecule) is not None for molecule in reaction.reactants)


def _merge_order(left, left_atom, right, right_atom) -> int | None:
    """Bond order closing two open ends, or None to refuse the pair.

    Two heteroatoms meeting at a cut are refused — the match that produced them
    was probably wrong.  Everything else joins with a single bond.
    """
    if (
        left.atom(left_atom).atomic_symbol in _NO_BOND
        and right.atom(right_atom).atomic_symbol in _NO_BOND
    ):
        return None
    return 1


def _merge(fragments) -> MoleculeContainer:
    """Close the open bonds of the missing fragments.

    Two fragments cut from their molecules are bonded to each other; a single
    one is capped on its own.  Intact fragments are carried over unchanged.
    """
    cut = [(fragment, boundary) for fragment, boundary in fragments if boundary]
    whole = [fragment for fragment, boundary in fragments if not boundary]
    # Compound rules. Water carried through untouched is a solvent,
    # not a product; an alcohol carried through beside a single open bond is
    # the reagent that closes it, so it is given an opening of its own.
    whole = [fragment for fragment in whole if fragment.brutto != {"O": 1, "H": 2}]
    if len(cut) == 1 and len(whole) == 1:
        oxygen = _alcohol_oxygen(whole[0])
        if oxygen is not None:
            cut.append((whole.pop(), (oxygen, None)))

    if len(cut) == 2:
        (left, (left_atom, _)), (right, (right_atom, _)) = cut
        order = _merge_order(left, left_atom, right, right_atom)
        if order is None:
            raise RebalancingError("refusing to bond two heteroatoms")
        merged = _join(left, right, left_atom, right_atom)
    elif len(cut) == 1:
        merged, (atom, cap) = cut[0]
        if cap is not None:
            element, order = cap
            merged.add_bond(atom, merged.add_atom(element), order)
    elif whole:
        merged = whole.pop()
    else:
        raise RebalancingError("nothing to impute")

    for fragment in whole:
        merged = merged.union(fragment, remap=True)
    return merged


def _match_formula(
    needed: dict[str, int], table=SMALL_MOLECULES, *, prefer_neutral: bool = False
) -> list[str]:
    """Cover ``needed`` exactly with molecules from :data:`SMALL_MOLECULES`.

    Searches depth first, taking the largest possible multiple of each molecule
    so that every step removes at least one element.  Among the solutions using
    the fewest molecules the most highly charged one wins.

    :param needed: Element counts to cover, net charge under ``"Q"``.
    :return: SMILES of the molecules to add, empty when nothing covers it.
    """
    solutions: list[list[str]] = []
    budget = [_MAX_SEARCH_NODES]

    def search(rest: dict[str, int], path: list[str]) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if not any(count for element, count in rest.items() if element != "Q"):
            if not rest.get("Q", 0):
                solutions.append(path)
            return
        for smiles_string, composition in table:
            counts = [
                rest.get(element, 0) // count
                for element, count in composition.items()
                if element != "Q" and count
            ]
            times = min(counts, default=0)
            if times <= 0:
                continue
            reduced = dict(rest)
            for element, count in composition.items():
                reduced[element] = reduced.get(element, 0) - count * times
            search(reduced, path + [smiles_string] * times)

    search(dict(needed), [])
    if not solutions:
        return []
    # A species used twice still counts once: what is being minimised is how
    # many different things the reaction is said to have lost, and among the
    # shortest the most ionic reading wins.
    shortest = min(len(set(solution)) for solution in solutions)
    charges = {name: abs(composition.get("Q", 0)) for name, composition in table}
    pool = [solution for solution in solutions if len(set(solution)) == shortest]
    # Completing a deficit with hydrogen means a neutral acid left the reaction;
    # an exact match without one means it was already a salt.
    pick = min if prefer_neutral else max
    return pick(pool, key=lambda solution: sum(charges[name] for name in solution))


def _small_molecules(smiles_strings: list[str]) -> list[MoleculeContainer]:
    """Parse the species to add.

    ``[O]`` is read with ``keep_implicit`` because chython otherwise fills the
    valence of a bare bracket atom and hands back water.
    """
    return [
        smiles_chython(text, keep_implicit=text == ATOMIC_OXYGEN)
        for text in smiles_strings
    ]


def _pair_protons(found: list[str]) -> list[str]:
    """Rewrite a proton beside an anion as the neutral molecule the two make."""
    found = list(found)
    while "[H+]" in found:
        anion = next((name for name in PROTON_PARTNERS if name in found), None)
        if anion is None:
            break
        found.remove("[H+]")
        found.remove(anion)
        found.append(PROTON_PARTNERS[anion])
    return found


def _oxidant_for_hydrogen(found: list[str]) -> tuple[list[str], list[str]]:
    """Read hydrogen given off as an oxidant taken up.

    A reaction does not vent dihydrogen: the hydrogen leaves on an oxygen the
    record never named, so each ``H2`` on the product side is written as the
    water it becomes, and the reactant side owes the oxygen atom for it.
    """
    rest = [name for name in found if name != "[H][H]"]
    couples = len(found) - len(rest)
    if not couples:
        return found, []
    return rest + ["O"] * couples, [ATOMIC_OXYGEN] * couples


def _cover(
    needed: dict[str, int],
    *,
    to_reactants: bool = False,
    leaving_acid: bool = False,
    spent: tuple = (),
) -> tuple[list[str], list[str]]:
    """Molecules covering ``needed``, and the hydrogens the other side owes.

    A deficit that names no hydrogen is usually a reagent written without one —
    a lone oxygen is really water.  Those are completed with hydrogens here,
    which the caller balances by putting the same count on the opposite side.

    :param needed: Element counts to cover, net charge under ``"Q"``.
    :param leaving_acid: The reaction already gave off a carbon fragment at a
        bond the mapping named, so a proton and an anion in the same deficit
        left together as one neutral molecule, not as a dissociated pair.  Only
        ever set for the product side; what goes in is weighed out as the salt.
    :return: SMILES to add, and the SMILES the opposite side owes for them.
    """
    found = _match_formula(needed, SMALL_MOLECULES + spent) if spent else []
    if not found:
        found = _match_formula(needed)
    if found:
        if leaving_acid:
            found = _pair_protons(found)
        if not to_reactants:
            return _oxidant_for_hydrogen(found)
        return found, []
    if (
        to_reactants
        and set(needed) <= {"O", "Q"}
        and not needed.get("Q")
        and needed.get("O", 0) > 0
    ):
        # An oxidant written as nothing but its oxygen, rather than water plus
        # the hydrogen the product side would then owe back.
        return [ATOMIC_OXYGEN] * needed["O"], []
    for hydrogens in (1, 2, 3, 4):
        found = _match_formula(
            {**needed, "H": needed.get("H", 0) + hydrogens},
            SMALL_MOLECULES + PROTONATED,
            prefer_neutral=True,
        )
        if found:
            return _neutralize(found), ["[H][H]"] * (hydrogens // 2) + ["[H]"] * (
                hydrogens % 2
            )
    # A metal keeps its valence by picking up hydroxide from an aqueous workup,
    # which the other side pays for in water.
    for hydroxides in (1, 2):
        found = _match_formula(
            {
                **needed,
                "O": needed.get("O", 0) + hydroxides,
                "H": needed.get("H", 0) + hydroxides,
            },
            SMALL_MOLECULES + LEAVING_GROUPS,
        )
        if found:
            return _neutralize(found), ["O"] * hydroxides
    return [], []


def _neutralize(found: list[str]) -> list[str]:
    """Put a borrowed hydrogen back on a hydroxide rather than beside it.

    Only reachable once a deficit is being completed with hydrogen: that extra
    hydrogen is what lets the formula match end on a proton and a hydroxide at
    once, and the pair is the water it was borrowed from.
    """
    while "[H+]" in found and "[OH-]" in found:
        found = list(found)
        found.remove("[H+]")
        found.remove("[OH-]")
        found.append("O")
    return found


def rebalance_reaction(
    reaction: ReactionContainer,
    *,
    add_redox_agents: bool = False,
    min_confidence: float = 0.0,
    drop_competing_products: bool = False,
    use_mapping: bool = True,
    refuse_unsupported_redox: bool = False,
) -> ReactionContainer:
    """Add the molecules an unbalanced reaction is missing.

    Carbon missing from the product side is recovered as substructures of the
    reactants; whatever imbalance is left over is covered with small molecules
    from :data:`SMALL_MOLECULES`.  A balanced reaction is returned unchanged.

    :param reaction: Input reaction, mapped or not.
    :param add_redox_agents: Name the reagent behind a plain loss of hydrogen
        instead of balancing it with loose hydrogen atoms.  This invents a
        species the record never mentioned, so it is off by default.
    :param min_confidence: Refuse an answer scoring below this under
        :func:`confidence`.  Zero, the default, answers whatever it can.
    :param drop_competing_products: First remove the products the reactants
        cannot have made, per :func:`competing_products`.  Off by default,
        since dropping a species the record listed is a larger claim than
        adding one it left out.
    :param refuse_unsupported_redox: Refuse an answer that balances by adding
        free hydrogen when the record names no reagent that could have moved
        it.  Judged on USPTO, those answers are right one time in five.
    :param use_mapping: Read the bonds that break off the CGR, worth ten
        points of accuracy on a hand-curated mapping.  Off is exactly
        equivalent to handing the same record over unmapped, since the CGR is
        the only thing the mapping is read for.  Turn it off for machine
        mapping you do not trust: a mapper handed an unbalanced reaction has
        atoms it cannot place, and a wrong CGR cuts in the wrong place.
    :return: The balanced reaction, with its score under ``meta["confidence"]``.
    :raises RebalancingError: If the reaction cannot be balanced, or the answer
        falls below ``min_confidence``.
    """
    if drop_competing_products:
        makeable = competing_products(reaction, add_redox_agents=add_redox_agents)
        if makeable is not None:
            if not makeable:
                raise RebalancingError("reactants can make none of the products")
            trimmed = ReactionContainer(
                reaction.reactants, makeable, reaction.reagents, reaction.meta
            )
            trimmed.name = reaction.name
            reaction = trimmed
    balanced = _rebalance_forward(
        reaction, add_redox_agents=add_redox_agents, use_mapping=use_mapping
    )
    if balanced is reaction:
        return balanced  # was balanced already; nothing was invented
    decomposed = ReactionContainer(
        _decompose_unstable(list(balanced.reactants)),
        _decompose_unstable(list(balanced.products)),
        balanced.reagents,
        balanced.meta,
    )
    decomposed.name = balanced.name
    balanced = decomposed
    _separate_numbering(balanced)
    if refuse_unsupported_redox:
        before = Counter(str(molecule) for molecule in reaction.molecules())
        after = Counter(str(molecule) for molecule in balanced.molecules())
        if any(
            set(smiles_chython(text).brutto) == {"H"}
            for text in (after - before).elements()
        ) and not _redox_is_recorded(reaction):
            raise RebalancingError("free hydrogen with no redox reagent recorded")
    score = confidence(reaction, balanced)
    if score < min_confidence:
        raise RebalancingError(f"confidence {score} below {min_confidence}")
    balanced.meta["confidence"] = score
    return balanced


def competing_products(
    reaction: ReactionContainer, *, add_redox_agents: bool = False
) -> list[MoleculeContainer] | None:
    """The products of a record that describes more than one reaction.

    A screening plate is read for everything on it: the product, the isomers
    competing with it, and an internal standard that took no part.  As one
    equation the reactants have to supply all of that at once, which balances
    only by declaring the extra products to be reactants.  The tell is carbon
    missing from the reactant side; products of a single transformation — an
    acid and the alcohol released beside it — leave the reactants able to
    cover the sum, and that record is not competing.

    Each product is then tried against the reactants alone.  The ones that
    balance are what the reactants could have made, one reaction each;
    whatever is left took no part.  Regioisomers share a formula, so a plate
    screening for two of them reports both.

    :param reaction: Input reaction, mapped or not.
    :param add_redox_agents: As :func:`rebalance_reaction`.
    :return: The products the reactants could have made, or ``None`` where
        nothing was competing.
    """
    if reaction_imbalance(reaction).get("C", 0) >= 0:
        return None
    makeable = []
    for product in reaction.products:
        candidate = ReactionContainer(reaction.reactants, [product])
        with suppress(RebalancingError):
            _rebalance_forward(candidate, add_redox_agents=add_redox_agents)
            makeable.append(product)
    return makeable


@cache
def _byproduct_entry(text: str) -> tuple[tuple[str, dict[str, int]], ...]:
    """The table entry for one reagent's spent form."""
    spent = REAGENT_BYPRODUCTS[text]
    return ((spent, dict(_formula([smiles_chython(spent)]))),)


def _spent_reagents(reaction: ReactionContainer) -> tuple:
    """Spent forms of the consumable reagents this record actually names."""
    names = {str(molecule) for molecule in reaction.molecules()}
    entries: tuple = ()
    for text in REAGENT_BYPRODUCTS:
        with suppress(Exception):
            if str(smiles_chython(text)) in names:
                entries += _byproduct_entry(text)
    return entries


def _decompose_unstable(molecules: list) -> list:
    """Replace an imputed species that cannot exist with what it falls apart into."""
    if not any(str(molecule) in UNSTABLE for molecule in molecules):
        return molecules
    out = []
    for molecule in molecules:
        replacement = UNSTABLE.get(str(molecule))
        out.extend(_small_molecules(list(replacement)) if replacement else [molecule])
    return out


def _redox_is_recorded(reaction: ReactionContainer) -> bool:
    """Whether the record names something that could have moved the hydrogen.

    Free hydrogen is the imputer's way of papering over a redox step it did
    not read, and it is wrong far more often than it is right.  A patent that
    actually reduced or oxidised something says so in its reagents — a
    palladium catalyst, a borohydride, a peroxide, elemental halogen.  With
    none of that recorded, inventing hydrogen invents the chemistry too.
    """
    for molecule in reaction.reagents:
        composition = molecule.brutto
        if _REDOX_ELEMENTS & set(composition):
            return True
        if set(composition) <= {"H"}:  # hydrogen itself was recorded
            return True
        if len(molecule) == 2 and set(composition) <= {"Br", "I", "Cl", "O"}:
            return True  # Br2, I2, O2 and the peroxides
        if any(
            bond.order == 1
            and molecule.atom(left).atomic_symbol == "O"
            and molecule.atom(right).atomic_symbol == "O"
            for left, right, bond in molecule.bonds()
        ):
            return True  # a peroxide oxidant
    return False


def _separate_numbering(reaction: ReactionContainer) -> None:
    """Give the imputed molecules atom numbers a CGR can be composed from.

    Imputed species are parsed fresh, so they start at 1 and land on top of the
    reaction's own numbering.  Two things then go wrong: one number names two
    atoms on the same side, and one number names a nitrogen among the reactants
    and an oxygen among the products.  chython refuses to compose a CGR from
    either, and every standardization step after this one composes one.  Only
    the atoms that actually clash move, so an imputed fragment keeps the
    reactant numbers that say where it came from.

    Reagents count as part of the reactant side: that is where
    ``ReactionContainer.compose`` puts them.
    """
    spare = max((n for m in reaction.molecules() for n in m), default=0)

    def renumber(molecule: MoleculeContainer, clash: set[int]) -> None:
        nonlocal spare
        molecule.remap({n: spare + i for i, n in enumerate(clash, 1)})
        spare += len(clash)

    left = list(reaction.reagents) + list(reaction.reactants)
    for side in (left, reaction.products):
        seen: set[int] = set()
        for molecule in side:
            clash = seen & set(molecule)
            if clash:
                renumber(molecule, clash)
            seen |= set(molecule)

    elements = {
        number: atom.atomic_symbol
        for molecule in left
        for number, atom in molecule.atoms()
    }
    for molecule in reaction.products:
        clash = {
            number
            for number, atom in molecule.atoms()
            if elements.get(number, atom.atomic_symbol) != atom.atomic_symbol
        }
        if clash:
            renumber(molecule, clash)


def _rebalance_forward(
    reaction: ReactionContainer,
    *,
    add_redox_agents: bool = False,
    use_mapping: bool = True,
) -> ReactionContainer:
    """:func:`rebalance_reaction` without the confidence gate."""

    imbalance = reaction_imbalance(reaction)
    if not imbalance:
        return reaction

    reactants = list(reaction.reactants)
    products = list(reaction.products)
    if (
        add_redox_agents
        and set(imbalance) == {"H"}
        and imbalance["H"] > 0
        and _is_alcohol_oxidation(reaction)
    ):
        # Nothing left the substrate but hydrogen: an oxidation whose oxidant
        # was never written down.
        units, remainder = divmod(imbalance["H"], 2)
        if not remainder:
            left, right = OXIDANT_COUPLE
            reactants.extend(_small_molecules(list(left) * units))
            products.extend(_small_molecules(list(right) * units))
            imbalance = {}
    if imbalance.get("C", 0) < 0:
        # Carbon short on the reactant side is the same problem seen from the
        # other end: solve the reaction backwards and turn the answer around.
        mirrored = _rebalance_forward(
            ReactionContainer(reaction.products, reaction.reactants),
            add_redox_agents=add_redox_agents,
            use_mapping=use_mapping,
        )
        balanced = ReactionContainer(
            mirrored.products, mirrored.reactants, reaction.reagents, reaction.meta
        )
        balanced.name = reaction.name
        return balanced
    # A halide left over beside a carbon fragment that was cut where the
    # mapping said the bond broke went with it, as the acid.
    leaving_acid = False
    if imbalance.get("C", 0) > 0:
        broken = _broken_bonds(reaction) if use_mapping else None
        products.extend(_missing_molecules(reaction, imbalance, broken).split())
        leaving_acid = broken is not None
        imbalance = reaction_imbalance(
            ReactionContainer(reactants, products, reaction.reagents)
        )

    # Oxygen left over on the reactant side while the products are short of
    # hydrogen is a reduction: the oxygen leaves as water and the hydrogen
    # arrives as hydrogen.  Balancing it literally would give off dioxygen.
    # Oxygen left over on the reactant side leaves as water: as the hydrogen
    # the products are short of when it is a reduction, and however the
    # hydrogen falls out when nothing but hydrogen and oxygen is missing.
    # Balancing it literally would give off dioxygen.
    if imbalance.get("O", 0) > 0 and (
        imbalance.get("H", 0) < 0 or set(imbalance) <= {"H", "O"}
    ):
        products.extend(_small_molecules(["O"] * imbalance["O"]))
        imbalance = reaction_imbalance(
            ReactionContainer(reactants, products, reaction.reagents)
        )

    if imbalance.get("O", 0) < 0 and any(
        count > 0
        for element, count in imbalance.items()
        if element not in ("O", "H", "Q")
    ):
        reactants.extend(_small_molecules(["O"] * -imbalance["O"]))
        imbalance = reaction_imbalance(
            ReactionContainer(reactants, products, reaction.reagents)
        )

    # Cover the product side first: a species completed with hydrogen or
    # hydroxide moves the reactant side too, so that one is settled against a
    # fresh count.  Paying for a completion can leave its own small remainder,
    # hence the second sweep.
    for _ in range(2):
        for side, other, sign in ((products, reactants, 1), (reactants, products, -1)):
            needed = {
                element: sign * count
                for element, count in imbalance.items()
                if element != "Q" and sign * count > 0
            }
            if not needed:
                continue
            # Charge travels with whichever side is short of atoms — split off
            # alone it would ask for a counter-ion the reaction never lost.
            needed["Q"] = sign * imbalance.get("Q", 0)
            # Only what leaves pairs up: a reagent the reactant side is short of
            # was written as the salt it was weighed out as.
            found, owed = _cover(
                needed,
                to_reactants=sign < 0,
                leaving_acid=leaving_acid and sign > 0,
                spent=_spent_reagents(reaction) if sign > 0 else (),
            )
            if not found:
                raise RebalancingError(f"no small molecules cover {needed}")
            if add_redox_agents and sign > 0 and "[H][H]" in found:
                # Hydrogen coming off is an oxidant going in: the reaction took
                # up the oxygen and gave back water.
                units = found.count("[H][H]")
                found = [name for name in found if name != "[H][H]"]
                found.extend(["O"] * units)
                reactants.extend(_small_molecules([ATOMIC_OXYGEN] * units))
            side.extend(_small_molecules(found))
            other.extend(_small_molecules(owed))
            imbalance = reaction_imbalance(
                ReactionContainer(reactants, products, reaction.reagents)
            )
        if not imbalance:
            break

    balanced = ReactionContainer(reactants, products, reaction.reagents, reaction.meta)
    balanced.name = reaction.name
    if reaction_imbalance(balanced):
        raise RebalancingError("reaction is still unbalanced")
    _reject_broken_valence(balanced, reaction)
    return balanced


def _reject_broken_valence(
    balanced: ReactionContainer, original: ReactionContainer
) -> None:
    """Refuse species invented with an unsatisfied valence.

    A metal cut out of its ligands is the case that matters: a CGR round trip
    happily yields "[Mg]Br", which is a divalent magnesium holding one bond.
    Carbon never trips this — chython caps it with hydrogen instead — so a
    truncated carbon fragment has to be caught when the break is chosen.
    """
    before = {str(molecule) for molecule in original.molecules()}
    for molecule in balanced.molecules():
        if str(molecule) in before:
            continue  # came in like that; not ours to reject
        if _has_broken_valence(molecule):
            raise RebalancingError(f"imputed {molecule} has an unsatisfied valence")
