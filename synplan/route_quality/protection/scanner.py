"""Route scanner for competing functional group interactions.

Walks a synthesis route step-by-step, detecting functional groups that
may compete with the intended reaction at each step, and classifies
their severity using an FG x FG incompatibility matrix.

The competing-sites identification approach is inspired by the methodology of:

    Westerlund et al., "Toward lab-ready AI synthesis plans with protection
    strategies and route scoring", *ChemRxiv*, 2025.
    https://doi.org/10.26434/chemrxiv-2025-gdrr8
"""

import csv
import logging

from chython.containers import ReactionContainer
from pydantic import BaseModel, ConfigDict

from synplan.route_quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.route_quality.protection.reaction_classifier import (
    get_reaction_center_atoms,
)

logger = logging.getLogger(__name__)

# Severity level mapping: integer code -> label
_SEVERITY_LABELS = {0: "compatible", 1: "competing", 2: "incompatible"}


class CompetingInteraction(BaseModel):
    """A competing functional group interaction at a synthesis step.

    :param step_id: Index of the reaction step in the route.
    :param fg_name: Name of the competing functional group.
    :param fg_atoms: Atom indices of the matched functional group.
    :param reacting_fg: Name of the FG at the reaction center (or None).
    :param severity: Interaction severity: "incompatible", "competing", or "compatible".
    """

    model_config = ConfigDict(frozen=True)

    step_id: int
    fg_name: str
    fg_atoms: tuple[int, ...]
    reacting_fg: str | None
    severity: str


class IncompatibilityMatrix:
    """Lookup table for functional group vs functional group incompatibility.

    Loads a TSV matrix where the first row contains column FG names
    (with an empty first cell) and subsequent rows have a row FG name
    followed by integer severity levels (0=compatible, 1=competing,
    2=incompatible).

    :param config_path: Path to the incompatibility matrix TSV file.
    """

    def __init__(self, config_path: str):
        self._matrix: dict[str, dict[str, int]] = {}
        with open(config_path, encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            col_names = header[1:]  # skip empty first cell
            for row in reader:
                row_name = row[0]
                self._matrix[row_name] = {
                    col_names[i]: int(row[i + 1])
                    for i in range(min(len(col_names), len(row) - 1))
                }

    def lookup(self, competing_fg: str, reacting_fg: str) -> str:
        """Look up the severity of a (competing_fg, reacting_fg) pair.

        :param competing_fg: Competing functional group name (row key).
        :param reacting_fg: Reacting functional group name (column key).
        :return: Severity label: "incompatible", "competing", or "compatible".
        """
        fg_row = self._matrix.get(competing_fg)
        if fg_row is None:
            return "compatible"
        level = fg_row.get(reacting_fg, 0)
        return _SEVERITY_LABELS.get(level, "compatible")


class RouteScanner:
    """Scan a synthesis route for competing functional group interactions.

    For each step in the route, detects functional groups on the product
    molecule that do not overlap with the reaction center, identifies the
    FG at the reaction center ("reacting FG"), and classifies their
    interaction severity using the FG x FG incompatibility matrix.
    Also counts same-family competing halogens for the H term.

    :param fg_detector: A FunctionalGroupDetector instance.
    :param incompatibility: An IncompatibilityMatrix instance.
    :param halogen_detector: An optional HalogenDetector instance.
    """

    def __init__(
        self,
        fg_detector: FunctionalGroupDetector,
        incompatibility: IncompatibilityMatrix,
        halogen_detector: HalogenDetector | None = None,
    ):
        self._fg_detector = fg_detector
        self._incompatibility = incompatibility
        self._halogen_detector = halogen_detector
        # Step-ids that ``scan_route`` actually processed on its last call
        # (i.e. whose CGR composed successfully). Consumers (e.g.
        # ``CompetingSitesScore``) use this to distinguish "scanned, no
        # issues" from "scan failed and silently skipped" when normalizing
        # the score by route length.
        self.last_processed_steps: set[int] = set()

    def scan_route(
        self, route: dict[int, ReactionContainer]
    ) -> tuple[list[CompetingInteraction], int]:
        """Walk a route step-by-step and collect competing interactions.

        For each step the scanner:

        1. Identifies the **reacting FG** — the FG on the *reactant* side
           that is consumed by the reaction (present in reactant, overlapping
           the reaction center).  This matches the paper's approach of
           looking at the FG being transformed.
        2. Identifies **competing FGs** on the *product* side that do not
           overlap the reaction center.
        3. Looks up severity of each competing FG against the reacting FG
           in the incompatibility matrix.

        :param route: A dict mapping step_id -> ReactionContainer, as
            returned by ``extract_reactions()`` in
            ``synplan.chem.reaction_routes.route_cgr``.
        :return: Tuple of (interactions, halogen_count) where interactions
            is a list of CompetingInteraction objects and halogen_count is
            the total number of same-family competing halogen sites.
        """
        interactions: list[CompetingInteraction] = []
        total_halogen_count = 0
        processed_steps: set[int] = set()

        for step_id in sorted(route):
            reaction = route[step_id]

            # 1. Get the product molecule (first product)
            products = list(reaction.products)
            if not products:
                continue
            product = products[0]

            # 2. Compose CGR once and reuse
            try:
                cgr = ~reaction
            except Exception:
                logger.warning("Could not compose CGR for step %d, skipping.", step_id)
                continue
            processed_steps.add(step_id)

            # 3. Find reaction center atoms (reuse CGR)
            center_atoms = get_reaction_center_atoms(reaction, cgr=cgr)

            # 4. Identify the reacting FG.
            #    Try the REACTANT side first (the consumed FG, per the paper),
            #    then fall back to the PRODUCT side (the formed FG).
            reacting_fg_name = None
            best_overlap = 0
            for reactant in reaction.reactants:
                reacting_match = self._fg_detector.detect_reacting(
                    reactant, center_atoms
                )
                if reacting_match is not None:
                    overlap = len(set(reacting_match.atom_indices) & center_atoms)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        reacting_fg_name = reacting_match.name

            # Fallback: check the product side if no reactant FG found
            if reacting_fg_name is None:
                product_match = self._fg_detector.detect_reacting(product, center_atoms)
                if product_match is not None:
                    reacting_fg_name = product_match.name

            # 5. Detect competing FGs on product (not overlapping reaction center)
            competing_fgs = self._fg_detector.detect_competing(product, center_atoms)

            # 6. Look up severity for each competing FG against reacting FG.
            #    If no reacting FG could be identified (reaction type not
            #    covered by our SMARTS library), default to "compatible"
            #    since the matrix has no information for unknown FG pairs.
            for fg in competing_fgs:
                if reacting_fg_name is not None:
                    severity = self._incompatibility.lookup(fg.name, reacting_fg_name)
                else:
                    severity = "compatible"
                interactions.append(
                    CompetingInteraction(
                        step_id=step_id,
                        fg_name=fg.name,
                        fg_atoms=fg.atom_indices,
                        reacting_fg=reacting_fg_name,
                        severity=severity,
                    )
                )

            # 7. Count same-family competing halogens
            if self._halogen_detector is not None:
                total_halogen_count += (
                    self._halogen_detector.count_same_family_competing(
                        product, center_atoms
                    )
                )

        self.last_processed_steps = processed_steps
        return interactions, total_halogen_count

    @staticmethod
    def classify_interactions(
        interactions: list[CompetingInteraction],
        halogen_count: int = 0,
    ) -> tuple[int, int, int]:
        """Count interactions by severity category.

        :param interactions: List of CompetingInteraction objects.
        :param halogen_count: Number of same-family competing halogen sites.
        :return: Tuple of (I, C, H) where:
            - I = number of incompatible interactions
            - C = number of competing interactions
            - H = number of same-family competing halogen sites
        """
        incompatible = 0
        competing = 0

        for inter in interactions:
            if inter.severity == "incompatible":
                incompatible += 1
            elif inter.severity == "competing":
                competing += 1

        return incompatible, competing, halogen_count
