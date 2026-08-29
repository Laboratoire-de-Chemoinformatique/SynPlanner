"""Functional group detection for protection strategy analysis.

SMARTS-based detection of reactive functional groups in molecules,
used to identify competing sites that may require protecting group
strategies during synthesis.
"""

import logging

import yaml
from chython import smarts
from chython.containers import MoleculeContainer
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


def _load_patterns(entries) -> list[dict]:
    """Compile ``(name, label, smarts)`` triples, dropping what chython rejects."""

    patterns = []
    for name, label, smarts_str in entries:
        try:
            query = smarts(smarts_str)
        except Exception as exc:
            logger.warning(
                "Could not parse SMARTS for %s (%s): %s", name, smarts_str, exc
            )
            continue
        patterns.append({"name": name, "label": label, "query": query})
    return patterns


def _distinct_matches(patterns, molecule, atoms_of) -> list[tuple]:
    """Every pattern match once, its atoms written by ``atoms_of``."""

    found, seen = [], set()
    for pattern in patterns:
        for mapping in pattern["query"].get_mapping(molecule):
            atoms = atoms_of(mapping.values())
            key = (pattern["name"], atoms)
            if key in seen:
                continue
            seen.add(key)
            found.append((pattern["name"], pattern["label"], atoms))
    return found


def _away_from_centre(matches: list, reaction_center_atoms: set[int]) -> list:
    """The matches that do not touch the reaction centre: the competing ones."""

    return [m for m in matches if not set(m.atom_indices) & reaction_center_atoms]


class FunctionalGroupMatch(BaseModel):
    """A single functional group match in a molecule.

    :param name: Human-readable name of the functional group (e.g. "hydroxyl").
    :param category: Reactivity category (e.g. "nucleophile", "electrophile").
    :param atom_indices: Tuple of matched atom indices in the molecule,
        sorted for deduplication.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    atom_indices: tuple[int, ...]


class FunctionalGroupDetector:
    """SMARTS-based functional group detector.

    Loads a YAML config of SMARTS patterns organized by category and
    uses chython substructure matching to detect functional groups in
    molecules.

    Results are cached by canonical SMILES (with hydrogens) so that
    repeated queries for the same molecule are fast.

    :param config_path: Path to a YAML file with SMARTS definitions,
        organized by category (nucleophile/electrophile/unsaturated).
    """

    def __init__(self, config_path: str):
        self._patterns: list[dict] = []
        self._templates: dict[str, tuple[str, str]] = {}
        # name, category and atoms as positions in the canonical SMILES order
        self._cache: dict[str, list[tuple[str, str, tuple[int, ...]]]] = {}
        self._load_config(config_path)

    def template_for(self, name: str) -> tuple[str, str] | None:
        """Return ``(template_smarts, modification_type)`` for a group.

        :param name: Functional-group name.
        :return: The mapped protection template and its modification type,
            or ``None`` when the library defines no template for ``name``.
        """
        return self._templates.get(name)

    def _load_config(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        entries = [
            (entry, category) for category, group in raw.items() for entry in group
        ]
        self._patterns = _load_patterns(
            (entry["name"], category, entry["smarts"]) for entry, category in entries
        )
        # Consumed by the protection strategy, not by detection.
        self._templates = {
            entry["name"]: (
                entry["template_smarts"],
                entry.get("modification_type", "label"),
            )
            for entry, _ in entries
            if entry.get("template_smarts")
        }

    def detect_all(self, molecule: MoleculeContainer) -> list[FunctionalGroupMatch]:
        """Detect all functional group matches in a molecule.

        Applies every loaded SMARTS pattern and returns deduplicated
        matches (unique by name + sorted atom indices).  Results are
        cached by canonical SMILES so that the same molecule is not
        re-scanned; this molecule's atom numbers are filled in on the way
        out, because the key carries none.

        :param molecule: A chython MoleculeContainer to search.
        :return: List of FunctionalGroupMatch objects.
        """
        order = molecule.smiles_atoms_order
        key = format(molecule, "h")
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cache[key] = self._scan(molecule, order)

        return [
            FunctionalGroupMatch(
                name=name,
                category=category,
                atom_indices=tuple(sorted(order[position] for position in positions)),
            )
            for name, category, positions in cached
        ]

    def _scan(
        self, molecule: MoleculeContainer, order: tuple[int, ...]
    ) -> list[tuple[str, str, tuple[int, ...]]]:
        """Match every pattern, atoms given as positions in ``order``."""
        position = {atom: index for index, atom in enumerate(order)}
        return _distinct_matches(
            self._patterns,
            molecule,
            lambda atoms: tuple(sorted(position[atom] for atom in atoms)),
        )

    def detect_competing(
        self,
        molecule: MoleculeContainer,
        reaction_center_atoms: set[int],
    ) -> list[FunctionalGroupMatch]:
        """Detect functional groups NOT overlapping with the reaction center.

        These are "competing" sites that may interfere with the intended
        reaction at the reaction center.

        :param molecule: A chython MoleculeContainer to search.
        :param reaction_center_atoms: Atom indices of the reaction center.
        :return: List of FunctionalGroupMatch objects for competing FGs.
        """
        return _away_from_centre(self.detect_all(molecule), reaction_center_atoms)

    def detect_reacting(
        self,
        molecule: MoleculeContainer,
        reaction_center_atoms: set[int],
    ) -> FunctionalGroupMatch | None:
        """Detect the functional group at the reaction center.

        Returns the first FG whose atoms overlap with the reaction center,
        or ``None`` if no known FG is found there.

        :param molecule: A chython MoleculeContainer to search.
        :param reaction_center_atoms: Atom indices of the reaction center.
        :return: The FunctionalGroupMatch at the reaction center, or None.
        """
        return next(
            (
                m
                for m in self.detect_all(molecule)
                if set(m.atom_indices) & reaction_center_atoms
            ),
            None,
        )

    def clear_cache(self) -> None:
        """Clear the internal results cache."""
        self._cache.clear()


class HalogenMatch(BaseModel):
    """A single halogen group match in a molecule.

    :param name: Name of the halogen pattern (e.g. "aryl_bromide").
    :param family: Halogen family (e.g. "bromide", "chloride").
    :param atom_indices: Tuple of matched atom indices in the molecule.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    family: str
    atom_indices: tuple[int, ...]


class HalogenDetector:
    """SMARTS-based halogen group detector.

    Loads a YAML config of halogen SMARTS patterns and detects halogens
    in molecules.  Used to count same-family competing halogens for
    the H term in the S(T) score.

    :param config_path: Path to a YAML file with halogen SMARTS definitions.
    """

    def __init__(self, config_path: str):
        self._patterns: list[dict] = []
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        self._patterns = _load_patterns(
            (name, entry["family"], entry["smarts"]) for name, entry in raw.items()
        )

    def detect_all(self, molecule: MoleculeContainer) -> list[HalogenMatch]:
        """Detect all halogen matches in a molecule.

        :param molecule: A chython MoleculeContainer to search.
        :return: List of HalogenMatch objects.
        """
        return [
            HalogenMatch(name=name, family=family, atom_indices=atoms)
            for name, family, atoms in _distinct_matches(
                self._patterns, molecule, lambda atoms: tuple(sorted(atoms))
            )
        ]

    def detect_competing_halogens(
        self,
        molecule: MoleculeContainer,
        reaction_center_atoms: set[int],
    ) -> list[HalogenMatch]:
        """Detect halogen groups NOT overlapping with the reaction center.

        :param molecule: A chython MoleculeContainer to search.
        :param reaction_center_atoms: Atom indices of the reaction center.
        :return: List of HalogenMatch objects for competing halogens.
        """
        return _away_from_centre(self.detect_all(molecule), reaction_center_atoms)

    def detect_reaction_center_halogens(
        self,
        molecule: MoleculeContainer,
        reaction_center_atoms: set[int],
    ) -> list[HalogenMatch]:
        """Detect halogen groups overlapping with the reaction center.

        :param molecule: A chython MoleculeContainer to search.
        :param reaction_center_atoms: Atom indices of the reaction center.
        :return: List of HalogenMatch objects at the reaction center.
        """
        return [
            m
            for m in self.detect_all(molecule)
            if set(m.atom_indices) & reaction_center_atoms
        ]

    def count_same_family_competing(
        self,
        molecule: MoleculeContainer,
        reaction_center_atoms: set[int],
    ) -> int:
        """Count competing halogens in the same family as reaction center halogens.

        Per the paper, only halogens at competing sites that share the
        same halogen family as a halogen at the reaction center count
        toward the H term in S(T).

        :param molecule: A chython MoleculeContainer to search.
        :param reaction_center_atoms: Atom indices of the reaction center.
        :return: Number of same-family competing halogen sites.
        """
        center_halogens = self.detect_reaction_center_halogens(
            molecule, reaction_center_atoms
        )
        if not center_halogens:
            return 0

        center_families = {h.family for h in center_halogens}
        competing_halogens = self.detect_competing_halogens(
            molecule, reaction_center_atoms
        )
        return sum(1 for h in competing_halogens if h.family in center_families)
