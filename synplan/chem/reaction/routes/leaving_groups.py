import contextlib

from chython.containers import CGRContainer, ReactionContainer
from chython.containers.bonds import DynamicBond
from chython.periodictable import At, DynamicElement


class Marked:
    """Mixin that adds a mark property and overrides isotope.

    Must be used together with an Element-based class (e.g. At) via
    multiple inheritance so the real atom behavior comes from Element.
    Uses __slots__ = () to avoid layout conflict with Element's slots.
    Concrete subclasses (MarkedAt) define the actual storage slot.
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mark = None
        self._isotope = 0

    @property
    def mark(self):
        return self._mark

    @mark.setter
    def mark(self, mark):
        self._mark = mark

    @property
    def isotope(self):
        return getattr(self, "_isotope", 0)

    @isotope.setter
    def isotope(self, value):
        self._isotope = int(value) if value is not None else 0

    def __repr__(self):
        return f"{self.symbol}({self.isotope})"

    @property
    def atomic_symbol(self) -> str:
        return self.__class__.__name__[6:]

    @property
    def symbol(self) -> str:
        return "X"

    def __len__(self):
        return super().__len__()


class MarkedAt(Marked, At):
    __slots__ = ("_mark",)
    atomic_number = At.atomic_number

    @property
    def atomic_symbol(self):
        return "At"

    @property
    def symbol(self):
        return "X"

    def __repr__(self):
        return f"X({self.isotope})"

    def __str__(self):
        return f"X({self.isotope})"

    def __hash__(self):
        return hash(
            (
                self.isotope,
                getattr(self, "atomic_number", 0),
                getattr(self, "charge", 0),
                getattr(self, "is_radical", False),
            )
        )


class DynamicX(DynamicElement):
    __slots__ = ("_isotope", "_mark")

    atomic_number = 85
    mass = 0.0
    group = 0
    period = 0
    isotopes_distribution = list(range(20))
    atomic_radius = 0.5
    isotopes_masses = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._isotope = None
        self._mark = None

    @property
    def mark(self):
        return getattr(self, "_mark", None)

    @mark.setter
    def mark(self, value):
        self._mark = value

    @property
    def isotope(self):
        return getattr(self, "_isotope", None)

    @isotope.setter
    def isotope(self, value):
        self._isotope = value

    @property
    def symbol(self) -> str:
        return "X"

    def valence_rules(
        self, charge: int = 0, is_radical: bool = False, valence: int = 0
    ) -> tuple:
        if (charge == 0 and not is_radical and (valence == 1)) or (
            charge == 0 and not is_radical and valence == 0
        ):
            return tuple()
        else:
            return tuple()

    def __repr__(self):
        return f"Dynamic{self.symbol}()"

    @property
    def p_charge(self) -> int:
        return self.charge

    @property
    def p_is_radical(self) -> bool:
        return self.is_radical

    @property
    def p_hybridization(self) -> int | None:
        return self.hybridization


# Route-level leaving-group operations on route-CGRs / synthon reactions,
# building on the Marked / MarkedAt / DynamicX primitives above.


def lg_process_reset(lg_cgr: CGRContainer, atom_num: int):
    """
    Normalize bonds in an extracted leaving group (X) fragment and flag the attachment atom as a radical.

    Scans all bonds in `lg_cgr`, converting any bond with undefined `p_order`
    but defined `order` into a `DynamicBond` of matching integer order. Then sets
    the atom at `atom_num` to a radical.

    Parameters
    ----------
    target_cgr : CGRContainer
        The CGR representing the isolated leaving-group fragment.
    atom_num : int
        Index of the attachment atom to mark as a radical.

    Returns
    -------
    CGRContainer
        The modified `lg_cgr` with normalized bonds and the specified atom
        flagged as a radical.
    """
    bond_items = list(lg_cgr._bonds.items())
    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            if bond.p_order is None and bond.order is not None:
                order = int(bond.order)
                lg_cgr.delete_bond(atom1, atom2)
                lg_cgr.add_bond(atom1, atom2, DynamicBond(order, order))
    lg_cgr._radicals[atom_num] = True
    return lg_cgr


def lg_replacer(route_cgr: CGRContainer):
    """
    Extract dynamic leaving-groups from a CGR and mark attachment points.

    Scans the input CGRContainer for bonds lacking explicit p_order (i.e., leaving-group attachments),
    severs those bonds, captures each leaving-group as its own CGRContainer, and inserts DynamicX
    markers at the attachment sites. Finally, reindexes the markers to ensure unique labels.

    Parameters
    ----------
    route_cgr : CGRContainer
        A CGR representing the full synthethic route.

    Returns
    -------
    synthon_cgr : CGRContainer
        The core synthon CGR with DynamicX atoms marking each former leaving-group site.
    lg_groups : dict[int, tuple[CGRContainer, int]]
        Mapping from each marker label to a tuple of:
        - the extracted leaving-group CGRContainer
        - the atom index where it was attached.
    """
    lg_groups = {}

    cgr_prods = [route_cgr.substructure(c) for c in route_cgr.connected_components]
    target_cgr = cgr_prods[0]

    bond_items = list(target_cgr._bonds.items())
    reaction = ReactionContainer.from_cgr(target_cgr)
    target_mol = reaction.products[0]
    max_in_target_mol = max(target_mol._atoms)

    k = 1
    atom_nums = []
    checked_atoms = set()

    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            if (
                bond.p_order is None
                and bond.order is not None
                and tuple(sorted([atom1, atom2])) not in checked_atoms
            ):
                if atom1 <= max_in_target_mol:
                    lg = DynamicX()
                    lg.mark = k
                    lg.isotope = k
                    order = bond.order
                    p_order = bond.p_order
                    target_cgr.delete_bond(atom1, atom2)
                    lg_cgrs = [
                        target_cgr.substructure(c)
                        for c in target_cgr.connected_components
                    ]
                    checked_atoms.add(tuple(sorted([atom1, atom2])))
                    if len(lg_cgrs) == 2:
                        lg_cgr = lg_cgrs[1]
                        lg_cgr = lg_process_reset(lg_cgr, atom2)
                        with contextlib.suppress(
                            ImportError, AttributeError, Exception
                        ):
                            lg_cgr.clean2d()
                    else:
                        continue
                    lg_groups[k] = (lg_cgr, atom2)
                    target_cgr = next(
                        iter(
                            target_cgr.substructure(c)
                            for c in target_cgr.connected_components
                        )
                    )
                    target_cgr.add_atom(lg, atom2)
                    if order == 4 and p_order is None:
                        order = 1
                    target_cgr.add_bond(atom1, atom2, DynamicBond(order, p_order))
                    target_cgr = next(
                        iter(
                            target_cgr.substructure(c)
                            for c in target_cgr.connected_components
                        )
                    )
                    k += 1
                    atom_nums.append(atom2)

    synthon_cgr = next(
        iter(target_cgr.substructure(c) for c in target_cgr.connected_components)
    )
    reaction = ReactionContainer.from_cgr(synthon_cgr)
    reactants = reaction.reactants

    atom_mark_map = {}  # To map atom numbers to their new marks
    g = 1
    for _n, r in enumerate(reactants):
        for atom_num in atom_nums:
            if atom_num in r._atoms:
                synthon_cgr._atoms[atom_num].mark = g
                atom_mark_map[atom_num] = g
                g += 1

    new_lg_groups = {}
    for original_mark in lg_groups:
        cgr_obj, a_num = lg_groups[original_mark]
        new_mark = atom_mark_map.get(a_num)
        if new_mark is not None:
            new_lg_groups[new_mark] = (cgr_obj, a_num)
    lg_groups = new_lg_groups

    return synthon_cgr, lg_groups


def lg_reaction_replacer(
    synthon_reaction: ReactionContainer, lg_groups: dict, max_in_target_mol: int
):
    """
    Replace marked leaving-groups (X) into synthon reactants.

    For each reactant in `synthon_reaction`, finds placeholder atoms
    (indices > `max_in_target_mol`) that match entries in `lg_groups`,
    replaces them with `MarkedAt` atoms labeled by their leaving-group key (X),
    and preserves original bond connectivity.

    Parameters
    ----------
    synthon_reaction : ReactionContainer
        Reaction containing reactants with X placeholders.
    lg_groups : dict[int, tuple[CGRContainer, int]]
        Mapping from X label to (X CGR, attachment atom index).
    max_in_target_mol : int
        Highest atom index of the core product; any atom_num above this is a placeholder.

    Returns
    -------
    List[Molecule]
        Reactant molecules with `MarkedAt` atoms reinserted at X attachment sites.
    """
    new_reactants = []
    for reactant in synthon_reaction.reactants:
        atom_keys = list(reactant._atoms.keys())
        for atom_num in atom_keys:
            if atom_num > max_in_target_mol:
                for k, val in lg_groups.items():
                    lg = MarkedAt()
                    if atom_num == val[1]:
                        lg.mark = k
                        lg.isotope = k
                        atom1 = next(iter(reactant._bonds[atom_num]))
                        bond = reactant._bonds[atom_num][atom1]
                        reactant.delete_bond(atom1, atom_num)
                        reactant.delete_atom(atom_num)
                        reactant.add_atom(lg, atom_num)
                        reactant.add_bond(atom1, atom_num, bond)
        new_reactants.append(reactant)
    return new_reactants


def all_lg_collect(subgroup):
    """
    Gather all leaving-group CGRContainers by route index.

    Scans `subgroup['routes_data']`, collects every CGRContainer per index,
    and returns a mapping from each index to the list of distinct containers.

    Parameters
    ----------
    subgroup : dict
        Must contain 'routes_data', a dict mapping pathway keys to
        dicts of {route_index: (CGRContainer, …)}.

    Returns
    -------
    dict[int, list[CGRContainer]]
        For each route index, a list of unique CGRContainer objects
        (duplicates by string are filtered out).
    """
    all_indices = set()
    for sub_dict in subgroup["routes_data"].values():
        all_indices.update(sub_dict.keys())

    # Dynamically initialize result and seen dictionaries
    result = {idx: [] for idx in all_indices}
    seen = {idx: set() for idx in all_indices}

    # Populate the result with unique CGRContainer objects
    for sub_dict in subgroup["routes_data"].values():
        for idx in sub_dict:
            cgr_container = sub_dict[idx][0]
            cgr_str = str(cgr_container)
            if cgr_str not in seen[idx]:
                seen[idx].add(cgr_str)
                result[idx].append(cgr_container)
    return result


def new_lg_reaction_replacer(synthon_reaction, new_lgs, max_in_target_mol):
    """
    Replace placeholder atom indices with marked leaving-group atoms in reactants.

    Iterates through each reactant in a `ReactionContainer`, finds atom indices
    corresponding to newly detached leaving-groups (those greater than the
    core’s maximum index), and replaces them with `MarkedAt` atoms bearing
    the correct X labels and isotopes. Bonds to the original attachment points
    are preserved.

    Parameters
    ----------
    synthon_reaction : ReactionContainer
        A reaction container whose `reactants` list contains molecules with
        dummy atoms (by index) marking where leaving-groups were removed.
    new_lgs : dict[int, int]
        Mapping from leaving-group label (int) to the atom index (int) in each
        reactant that should be replaced.
    max_in_target_mol : int
        The highest atom index used by the core product. Any atom index in a
        reactant greater than this is treated as a leaving-group placeholder.

    Returns
    -------
    List[Molecule]
        A list of reactant molecules where each placeholder atom has been
        replaced by a `MarkedAt` atom with its `.mark` and `.isotope` set
        to the leaving-group label, and original bonds reattached.
    """
    new_reactants = []
    for reactant in synthon_reaction.reactants:
        atom_keys = list(reactant._atoms.keys())
        for atom_num in atom_keys:
            if atom_num > max_in_target_mol:
                for k, val in new_lgs.items():
                    lg = MarkedAt()
                    if atom_num == val:
                        lg.mark = k
                        lg.isotope = k
                        atom1 = next(iter(reactant._bonds[atom_num]))
                        bond = reactant._bonds[atom_num][atom1]
                        reactant.delete_bond(atom1, atom_num)
                        reactant.delete_atom(atom_num)
                        reactant.add_atom(lg, atom_num)
                        reactant.add_bond(atom1, atom_num, bond)
        new_reactants.append(reactant)

    return new_reactants
