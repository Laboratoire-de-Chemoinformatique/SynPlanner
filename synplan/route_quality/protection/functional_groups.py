"""Functional group detection for protection strategy analysis.

SMARTS-based detection of reactive functional groups in molecules,
used to identify competing sites that may require protecting group
strategies during synthesis.
"""

import logging

import yaml
from pydantic import BaseModel, ConfigDict
from chython import smarts
from chython.containers import MoleculeContainer

logger = logging.getLogger(__name__)


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
        self._cache: dict[str, list[FunctionalGroupMatch]] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        for category, entries in raw.items():
            for entry in entries:
                name = entry["name"]
                smarts_str = entry["smarts"]
                try:
                    query = smarts(smarts_str)
                except Exception as exc:
                    logger.warning(
                        "Could not parse SMARTS for %s (%s): %s",
                        name,
                        smarts_str,
                        exc,
                    )
                    continue
                self._patterns.append(
                    {
                        "name": name,
                        "category": category,
                        "query": query,
                    }
                )

    def _cache_key(self, molecule: MoleculeContainer) -> str:
        """Return a canonical SMILES string suitable as a cache key."""
        return format(molecule, "h")

    def detect_all(self, molecule: MoleculeContainer) -> list[FunctionalGroupMatch]:
        """Detect all functional group matches in a molecule.

        Applies every loaded SMARTS pattern and returns deduplicated
        matches (unique by name + sorted atom indices).  Results are
        cached by canonical SMILES so that the same molecule is not
        re-scanned.

        :param molecule: A chython MoleculeContainer to search.
        :return: List of FunctionalGroupMatch objects.
        """
        key = self._cache_key(molecule)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        matches: list[FunctionalGroupMatch] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()

        for pat in self._patterns:
            query = pat["query"]
            for mapping in query.get_mapping(molecule):
                atom_indices = tuple(sorted(mapping.values()))
                dedup_key = (pat["name"], atom_indices)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                matches.append(
                    FunctionalGroupMatch(
                        name=pat["name"],
                        category=pat["category"],
                        atom_indices=atom_indices,
                    )
                )

        self._cache[key] = matches
        return matches

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
        all_matches = self.detect_all(molecule)
        return [
            m for m in all_matches if not set(m.atom_indices) & reaction_center_atoms
        ]

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
        all_matches = self.detect_all(molecule)
        for m in all_matches:
            if set(m.atom_indices) & reaction_center_atoms:
                return m
        return None

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

        for name, entry in raw.items():
            smarts_str = entry["smarts"]
            family = entry["family"]
            try:
                query = smarts(smarts_str)
            except Exception as exc:
                logger.warning(
                    "Could not parse halogen SMARTS for %s (%s): %s",
                    name,
                    smarts_str,
                    exc,
                )
                continue
            self._patterns.append(
                {
                    "name": name,
                    "family": family,
                    "query": query,
                }
            )

    def detect_all(self, molecule: MoleculeContainer) -> list[HalogenMatch]:
        """Detect all halogen matches in a molecule.

        :param molecule: A chython MoleculeContainer to search.
        :return: List of HalogenMatch objects.
        """
        matches: list[HalogenMatch] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()

        for pat in self._patterns:
            query = pat["query"]
            for mapping in query.get_mapping(molecule):
                atom_indices = tuple(sorted(mapping.values()))
                key = (pat["name"], atom_indices)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    HalogenMatch(
                        name=pat["name"],
                        family=pat["family"],
                        atom_indices=atom_indices,
                    )
                )

        return matches

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
        all_matches = self.detect_all(molecule)
        return [
            m for m in all_matches if not set(m.atom_indices) & reaction_center_atoms
        ]

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
        all_matches = self.detect_all(molecule)
        return [m for m in all_matches if set(m.atom_indices) & reaction_center_atoms]

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
