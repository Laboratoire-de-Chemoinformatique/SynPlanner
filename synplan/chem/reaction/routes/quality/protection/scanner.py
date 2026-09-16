"""Route scanner for competing functional group interactions.

Walks a synthesis route step-by-step, detecting functional groups that
may compete with the intended reaction at each step, and classifies
their severity using an FG x FG incompatibility matrix.

The competing-sites identification approach is inspired by the methodology of:

    Westerlund et al., "Toward lab-ready AI synthesis plans with protection
    strategies and route scoring", *J. Chem. Inf. Model.* 2026, 66, 6361.
    https://doi.org/10.1021/acs.jcim.6c01147
"""

import csv
import logging
from typing import TYPE_CHECKING

from chython import smarts
from chython.containers import ReactionContainer
from pydantic import BaseModel, ConfigDict

from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.chem.reaction.routes.quality.protection.reaction_classifier import (
    get_reaction_center_atoms,
)

if TYPE_CHECKING:
    from synplan.chem.reaction.routes.route import Route

logger = logging.getLogger(__name__)

# Severity level mapping: integer code -> label
_SEVERITY_LABELS = {0: "compatible", 1: "competing", 2: "incompatible"}
#: No entry for the pair. Scored like "compatible" because there is nothing to score,
#: but named apart so a clean report is not mistaken for a checked one.
UNKNOWN_SEVERITY = "unknown"


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
    anchor_atom: int | None = None
    center_atoms: tuple[int, ...] = ()
    product_index: int = 0
    modification_type: str | None = None
    template_smarts: str | None = None
    site_key: str | None = None


class CompetingScanResult(BaseModel):
    """Interactions and explicit coverage, with this route's mapped site handles."""

    model_config = ConfigDict(frozen=True)

    interactions: list[CompetingInteraction]
    halogen_count: int
    processed_steps: tuple[int, ...]
    skipped_steps: tuple[int, ...]


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
            reader = csv.DictReader(fh, delimiter="\t")
            names = reader.fieldnames[0]  # the header's empty first cell
            for row in reader:
                self._matrix[row[names]] = {
                    column: int(severity)
                    for column, severity in row.items()
                    if column != names and column is not None and severity is not None
                }

    def lookup(self, competing_fg: str, reacting_fg: str) -> str:
        """Look up the severity of a (competing_fg, reacting_fg) pair.

        :param competing_fg: Competing functional group name (row key).
        :param reacting_fg: Reacting functional group name (column key).
        The matrix is dense, so a stored 0 means the pair was assessed and is fine.
        A missing row or column means nobody assessed it, which is not the same thing.

        :return: "incompatible", "competing", "compatible", or "unknown".
        """
        fg_row = self._matrix.get(competing_fg)
        if fg_row is None or reacting_fg not in fg_row:
            return UNKNOWN_SEVERITY
        return _SEVERITY_LABELS.get(fg_row[reacting_fg], UNKNOWN_SEVERITY)


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
        self._template_queries: dict[str, object] = {}

    def scan_route(
        self,
        route: "Route | dict[int, ReactionContainer]",
        *,
        detailed: bool = False,
    ) -> tuple[list[CompetingInteraction], int] | CompetingScanResult:
        """Scan all products, using ``Step.product`` for reaction classification.

        Dict callers retain their first-product convention. Detailed scans attach
        exact template anchors and explicitly report incomplete coverage. Cached
        functional-group matches are translated to each molecule's atom numbers.
        """
        from synplan.chem.reaction.routes.route import Route

        if isinstance(route, Route):
            reactions = route.reactions_dict
            selected = {i: step.product for i, step in enumerate(route)}
        else:
            reactions = route
            selected = {}
        interactions = []
        total_halogen_count = 0
        processed_steps = set()
        for step_id in sorted(reactions):
            reaction = reactions[step_id]
            products = list(reaction.products)
            if not products:
                continue
            product = selected.get(step_id, products[0])
            try:
                cgr = ~reaction
                center_atoms = get_reaction_center_atoms(reaction, cgr=cgr)
            except Exception:
                logger.warning("Could not compose CGR for step %d, skipping.", step_id)
                continue
            processed_steps.add(step_id)
            # Identify the reacting FG on the reactant side first.
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

            # Fallback: check the product side if no reactant FG found.
            if reacting_fg_name is None:
                product_match = self._fg_detector.detect_reacting(product, center_atoms)
                if product_match is not None:
                    reacting_fg_name = product_match.name

            for product_index, current in enumerate(products):
                for fg in self._fg_detector.detect_competing(current, center_atoms):
                    severity = (
                        self._incompatibility.lookup(fg.name, reacting_fg_name)
                        if reacting_fg_name is not None
                        else UNKNOWN_SEVERITY
                    )
                    extra = {}
                    if detailed:
                        template = self._fg_detector.template_for(fg.name)
                        anchor = None
                        if template is not None:
                            pattern, modification = template
                            if pattern not in self._template_queries:
                                try:
                                    self._template_queries[pattern] = smarts(pattern)
                                except ValueError:
                                    self._template_queries[pattern] = None
                            query = self._template_queries[pattern]
                            anchors = {
                                mapping[1]
                                for mapping in (
                                    () if query is None else query.get_mapping(current)
                                )
                                if mapping.get(1) in fg.atom_indices
                            }
                            if len(anchors) == 1:
                                anchor = next(iter(anchors))
                            extra.update(
                                template_smarts=pattern, modification_type=modification
                            )
                        extra.update(
                            anchor_atom=anchor,
                            center_atoms=tuple(sorted(center_atoms)),
                            site_key=f"{step_id}:{product_index}:{fg.name}:"
                            + ",".join(map(str, fg.atom_indices)),
                        )
                    interactions.append(
                        CompetingInteraction(
                            step_id=step_id,
                            fg_name=fg.name,
                            fg_atoms=fg.atom_indices,
                            reacting_fg=reacting_fg_name,
                            severity=severity,
                            product_index=product_index,
                            **extra,
                        )
                    )
                if self._halogen_detector is not None:
                    total_halogen_count += (
                        self._halogen_detector.count_same_family_competing(
                            current,
                            center_atoms,
                        )
                    )
        self.last_processed_steps = processed_steps
        if detailed:
            return CompetingScanResult(
                interactions=interactions,
                halogen_count=total_halogen_count,
                processed_steps=tuple(sorted(processed_steps)),
                skipped_steps=tuple(sorted(set(reactions) - processed_steps)),
            )
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
