"""Protection-group strategy: choose a protecting group and splice it into a route.

Implements the corrective half of Westerlund et al., *ChemRxiv*, 2025
(https://doi.org/10.26434/chemrxiv-2025-gdrr8): the scanner reports competing
sites, this module decides what to protect, builds the protection /
deprotection pair, and rewrites the route around the affected steps.

The paper ranks candidate groups with a Chemformer classifier whose weights are
not shipped here. :class:`ProtectingGroupClassifier` is that wiring point and
predicts nothing; the default :class:`FirstAllowedClassifier` keeps the order
the allowed-label mapping gives, which is a rule-only fallback rather than a
reproduction of the paper.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from chython import smarts, smiles
from chython.containers import MoleculeContainer, QueryContainer, ReactionContainer

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction.routes.quality.protection.scanner import (
    CompetingInteraction,
)

logger = logging.getLogger(__name__)

#: Atom map number marking the attachment atom inside a protecting-group template.
_PG_ATTACHMENT_MAP = 512

#: Atom map number marking the protected site inside a functional-group template.
_SITE_MAP = 1


@dataclass(frozen=True)
class ProtectingGroup:
    """One row of the protecting-group template table.

    :param label: Numeric id referenced by the allowed-label mapping.
    :param reaction_class: Named reaction class of the protection step.
    :param template: SMARTS fragment of the group, whose attachment atom
        carries map number 512.
    :param example_reagent: SMILES of a reagent introducing the group.
    :param substructure: SMILES of the attached group.
    :param deprotection_class: Named reaction class of the deprotection step.
    """

    label: int
    reaction_class: str
    template: str
    example_reagent: str
    substructure: str
    deprotection_class: str


def load_protecting_groups(path: str | Path) -> dict[int, list[ProtectingGroup]]:
    """Load the protecting-group template table, grouped by label.

    A label maps to several rows when one group has more than one template
    (six of the labels do), so every row is kept and the choice between them is
    left to the classifier.

    Rows whose template does not parse are dropped here rather than failing
    once per call site. Nine of them map two atoms to :512 to attach at two
    points, which acetals and dithianes need because they replace the carbonyl
    double bond rather than hanging a group off it; the product builder adds a
    single bond and deletes nothing, so it cannot express them. The table is
    kept byte-identical to the published one and the gap lives in the code.

    :param path: Path to the tab-separated template table.
    :return: Mapping of label id to its :class:`ProtectingGroup` rows, in file
        order.
    """
    groups: dict[int, list[ProtectingGroup]] = {}
    unsupported: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            label = int(row["label"])
            try:
                smarts(row["template"])
            except Exception as err:
                unsupported.setdefault(label, str(err))
                continue
            groups.setdefault(label, []).append(
                ProtectingGroup(
                    label=label,
                    reaction_class=row["reaction_class"],
                    template=row["template"],
                    example_reagent=row["example_reagent"],
                    substructure=row["substructure"],
                    deprotection_class=row["deprotection_class"],
                )
            )
    if unsupported:
        logger.warning(
            "Protecting-group labels %s have no usable template, so the "
            "functional groups that allow only those labels cannot be "
            "protected: %s",
            sorted(unsupported),
            "; ".join(f"{label}: {err}" for label, err in sorted(unsupported.items())),
        )
    return groups


def load_allowed_labels(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load the reactive-function to allowed-label mapping.

    The file is a pandas ``table``-orient dump; only its ``data`` records are
    read. Keys are the same functional-group names used as row and column keys
    of the incompatibility matrix.

    :param path: Path to the label-mapping JSON file.
    :return: Mapping of functional-group name to allowed label ids.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        record["reactive_function"]: tuple(record["labels"])
        for record in payload["data"]
    }


class ProtectingGroupClassifier:
    """Rank candidate protecting groups for one site.

    This is the seam for the paper's Chemformer ranking model. The base
    implementation deliberately predicts nothing so that an unwired
    installation degrades to the rule-only path instead of guessing; wire a
    real model in by subclassing and returning candidates in preference order.
    """

    def rank(
        self,
        fg_name: str,
        molecule: MoleculeContainer,
        site_atoms: tuple[int, ...],
        candidates: list[ProtectingGroup],
    ) -> list[ProtectingGroup]:
        """Return ``candidates`` in preference order, best first.

        :param fg_name: Name of the functional group being protected.
        :param molecule: Molecule carrying the site.
        :param site_atoms: Atom indices of the matched functional group.
        :param candidates: Groups allowed for ``fg_name`` by the mapping.
        :return: Empty list — the base model makes no prediction.
        """
        return []


class FirstAllowedClassifier(ProtectingGroupClassifier):
    """Rule-only fallback keeping the order of the allowed-label mapping."""

    def rank(
        self,
        fg_name: str,
        molecule: MoleculeContainer,
        site_atoms: tuple[int, ...],
        candidates: list[ProtectingGroup],
    ) -> list[ProtectingGroup]:
        """Return the allowed candidates unchanged."""
        return list(candidates)


@dataclass(frozen=True)
class ProtectionAction:
    """A decision to protect one competing site with one group.

    :param step_id: Step whose competing site is being protected.
    :param fg_name: Name of the competing functional group.
    :param site_atoms: Atom indices of the matched functional group.
    :param group: The chosen protecting group.
    """

    step_id: int
    fg_name: str
    site_atoms: tuple[int, ...]
    group: ProtectingGroup


class ProtectionPlanner:
    """Choose a protecting group for each competing site worth protecting.

    :param groups: Protecting-group table from :func:`load_protecting_groups`.
    :param allowed_labels: Mapping from :func:`load_allowed_labels`.
    :param classifier: Ranking model; defaults to the rule-only fallback.
    :param protect_severities: Severities that trigger protection.
    """

    def __init__(
        self,
        groups: dict[int, list[ProtectingGroup]],
        allowed_labels: dict[str, tuple[int, ...]],
        classifier: ProtectingGroupClassifier | None = None,
        protect_severities: tuple[str, ...] = ("incompatible",),
    ):
        self._groups = groups
        self._allowed = allowed_labels
        self._classifier = classifier or FirstAllowedClassifier()
        self._protect_severities = protect_severities

    def candidates(self, fg_name: str) -> list[ProtectingGroup]:
        """Return the groups the mapping allows for a functional group.

        :param fg_name: Functional-group name.
        :return: Allowed groups, in mapping order; empty if none apply.
        """
        return [
            group
            for label in self._allowed.get(fg_name, ())
            for group in self._groups.get(label, ())
        ]

    def plan(
        self,
        interactions: list[CompetingInteraction],
        molecules: dict[int, MoleculeContainer] | None = None,
    ) -> list[ProtectionAction]:
        """Choose one protecting group per competing site worth protecting.

        Sites whose functional group has no allowed group, and sites the
        classifier declines to rank, are skipped rather than guessed at.

        :param interactions: Competing interactions from ``RouteScanner``.
        :param molecules: Optional step_id to molecule map passed to the
            classifier for context.
        :return: One :class:`ProtectionAction` per protected site.
        """
        molecules = molecules or {}
        actions: list[ProtectionAction] = []
        for interaction in interactions:
            if interaction.severity not in self._protect_severities:
                continue
            allowed = self.candidates(interaction.fg_name)
            if not allowed:
                logger.debug(
                    "No protecting group allowed for %s at step %d.",
                    interaction.fg_name,
                    interaction.step_id,
                )
                continue
            ranked = self._classifier.rank(
                interaction.fg_name,
                molecules.get(interaction.step_id),
                interaction.fg_atoms,
                allowed,
            )
            if not ranked:
                logger.debug(
                    "Classifier ranked no group for %s at step %d.",
                    interaction.fg_name,
                    interaction.step_id,
                )
                continue
            actions.append(
                ProtectionAction(
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_atoms=interaction.fg_atoms,
                    group=ranked[0],
                )
            )
        return actions


def _fused_product(fg_template: str, pg_template: str) -> QueryContainer:
    """Build the product pattern: the site with the group bonded to it.

    ``smarts`` numbers atoms by their map, so the site keeps number 1 and the
    group's attachment atom keeps 512; the group's unmapped atoms are renumbered
    on merge. Fusing the containers directly avoids composing a SMARTS string,
    which chython cannot parse across a disconnection.

    :param fg_template: Mapped SMARTS of the functional group.
    :param pg_template: Mapped SMARTS of the protecting group.
    :return: The product query.
    """
    product = smarts(fg_template)
    if _SITE_MAP not in {n for n, _ in product.atoms()}:
        raise ValueError(f"No atom mapped :{_SITE_MAP} in {fg_template!r}")

    group_query = smarts(pg_template)
    renumbered: dict[int, int] = {}
    for number, atom in group_query.atoms():
        renumbered[number] = product.add_atom(atom.copy())
    for atom, neighbour, bond in group_query.bonds():
        product.add_bond(renumbered[atom], renumbered[neighbour], bond.copy())

    if _PG_ATTACHMENT_MAP not in renumbered:
        raise ValueError(f"No atom mapped :{_PG_ATTACHMENT_MAP} in {pg_template!r}")
    product.add_bond(_SITE_MAP, renumbered[_PG_ATTACHMENT_MAP], 1)
    return product


def build_protection_rule(
    fg_template: str, group: ProtectingGroup
) -> CanonicalRetroReactor:
    """Build a reactor attaching ``group`` at a functional-group site.

    :param fg_template: Mapped SMARTS of the functional group, from the
        ``template_smarts`` field of the competing-groups library.
    :param group: The protecting group to attach.
    :return: A :class:`~synplan.chem.reaction.CanonicalRetroReactor` applying
        the protection, so products come back canonical.
    """
    return CanonicalRetroReactor(
        patterns=(smarts(fg_template),),
        products=(_fused_product(fg_template, group.template),),
        delete_atoms=False,
    )


def protect_molecule(
    molecule: MoleculeContainer, fg_template: str, group: ProtectingGroup
) -> MoleculeContainer | None:
    """Attach ``group`` to the first matching site of ``molecule``.

    :param molecule: Molecule carrying the competing site.
    :param fg_template: Mapped SMARTS of the functional group.
    :param group: The protecting group to attach.
    :return: The protected molecule, or ``None`` when the template does not
        apply or the product cannot be built.
    """
    try:
        reactor = build_protection_rule(fg_template, group)
    except Exception as err:
        logger.warning("Could not build protection rule for %s: %s", group.label, err)
        return None
    try:
        for products in apply_reaction_rule(molecule, reactor, top_reactions_num=1):
            if products:
                return products[0]
    except Exception as err:
        logger.warning("Protection rule failed on molecule: %s", err)
    return None


def protection_pair(
    molecule: MoleculeContainer, fg_template: str, group: ProtectingGroup
) -> tuple[ReactionContainer, ReactionContainer] | None:
    """Build the protection and deprotection reactions for one site.

    :param molecule: Molecule carrying the competing site.
    :param fg_template: Mapped SMARTS of the functional group.
    :param group: The protecting group to attach.
    :return: ``(protection, deprotection)``, or ``None`` if protection fails.
    """
    protected = protect_molecule(molecule, fg_template, group)
    if protected is None:
        return None
    reactants = [molecule]
    if group.example_reagent:
        try:
            reagent = smiles(group.example_reagent)
            reagent.canonicalize()
            reactants.append(reagent)
        except Exception as err:
            logger.warning(
                "Could not parse reagent %r for label %s: %s",
                group.example_reagent,
                group.label,
                err,
            )

    protection = ReactionContainer(reactants=reactants, products=[protected])
    protection.meta["reaction_class"] = group.reaction_class
    deprotection = ReactionContainer(reactants=[protected], products=[molecule])
    deprotection.meta["reaction_class"] = group.deprotection_class
    return protection, deprotection


def apply_protection(
    route: dict[int, ReactionContainer],
    protection: ReactionContainer,
    deprotection: ReactionContainer,
    first_step: int,
    last_step: int,
) -> dict[int, ReactionContainer]:
    """Splice a protection/deprotection pair around a range of route steps.

    Steps are renumbered contiguously: protection is inserted immediately
    before ``first_step`` and deprotection immediately after ``last_step``,
    leaving the original reactions untouched.

    :param route: A ``step_id -> ReactionContainer`` route.
    :param protection: The protection reaction.
    :param deprotection: The deprotection reaction.
    :param first_step: Step the protection must precede.
    :param last_step: Step the deprotection must follow.
    :return: A new route dict with two extra steps.
    :raises KeyError: If either step is not in the route.
    :raises ValueError: If ``last_step`` comes before ``first_step``.
    """
    missing = {first_step, last_step} - set(route)
    if missing:
        # the splice is positional, so an absent step would drop its half of the
        # pair and hand back a route that looks protected and is not
        raise KeyError(f"steps {sorted(missing)} are not in the route")
    if last_step < first_step:
        raise ValueError(
            f"last_step {last_step} precedes first_step {first_step}, which "
            "would deprotect before protecting"
        )

    spliced: dict[int, ReactionContainer] = {}
    next_id = 0
    for step_id in sorted(route):
        if step_id == first_step:
            spliced[next_id] = protection
            next_id += 1
        spliced[next_id] = route[step_id]
        next_id += 1
        if step_id == last_step:
            spliced[next_id] = deprotection
            next_id += 1
    return spliced


__all__ = [
    "FirstAllowedClassifier",
    "ProtectingGroup",
    "ProtectingGroupClassifier",
    "ProtectionAction",
    "ProtectionPlanner",
    "apply_protection",
    "build_protection_rule",
    "load_allowed_labels",
    "load_protecting_groups",
    "protect_molecule",
    "protection_pair",
]
