from __future__ import annotations
from pathlib import Path

import pytest
from CGRtools import smiles
from CGRtools.containers import ReactionContainer, CGRContainer

from synplan.chem.data.filtering import (
    ReactionFilterConfig,
    CCsp3BreakingConfig,
    DynamicBondsConfig,
)
from synplan.chem.data.standardizing import (
    ReactionStandardizationConfig,
    ReactionMappingConfig,
    FunctionalGroupsConfig,
    KekuleFormConfig,
    CheckValenceConfig,
    ImplicifyHydrogensConfig,
    CheckIsotopesConfig,
    AromaticFormConfig,
    MappingFixConfig,
    UnchangedPartsConfig,
    RemoveReagentsConfig,
    RebalanceReactionConfig,
    DuplicateReactionConfig,
)
from synplan.chem.reaction_rules.extraction import RuleExtractionConfig


# ---------- test data ------------------------------------------------------ #

REACTIONS = [
    (
        "[CH3:5][CH2:6][OH:7].[O:4]=[C:2]([OH:3])[CH3:1]>[O:8]=[S:9](=[O:10])([OH:11])[OH:12]>[CH3:5][CH2:6][O:7][C:2](=[O:4])[CH3:1].[OH2:3]",
        "Fischer esterification",
    ),
    (
        "[CH2:5]=[CH2:6].[CH:2]([CH:3]=[CH2:4])=[CH2:1]>>[CH:3]1=[CH:4][CH2:6][CH2:5][CH2:2][CH2:1]1",
        "Diels–Alder cycloaddition",
    ),
    (
        "[CH:2](=[O:3])[CH3:1].[CH:5](=[O:6])[CH3:4]>[Na+:7].[OH-:8]>[OH:3][CH:2]([CH2:4][CH:5]=[O:6])[CH3:1]",
        "Aldol addition",
    ),
    (
        "[C-:3]#[N:4].[CH3:1][Br:2].[Na+:5]>CN(C)C=O>[Na+:5].[Br-:2].[N:4]#[C:3][CH3:1]",
        "SN2 substitution",
    ),
    (
        "[CH:2]1=[CH:3][CH2:4][CH2:5][CH2:6][CH2:1]1>[Pd:7]>[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH2:6]1",
        "Catalytic hydrogenation",
    ),
    (
        "[Cl:8][CH3:7].[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>[Cl:9][Al:10]([Cl:11])[Cl:12]>[ClH:8].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[CH3:7]",
        "Friedel–Crafts alkylation",
    ),
    (
        "[O:9]=[C:8]([CH3:7])[Cl:10].[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>[Cl:11][Al:12]([Cl:13])[Cl:14]>[ClH:10].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[C:8](=[O:9])[CH3:7]",
        "Friedel–Crafts acylation",
    ),
    (
        "[CH3:9][C:10]([CH3:12])=[O:11].[Mg:8].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[Br:7]>[CH3:18][CH2:17][O:16][CH2:15][CH3:14]>[Br-:7].[Mg+2:13].[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[C:10](=[O:11])[CH2:9][CH3:12]",
        "Grignard addition",
    ),
    (
        "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH:6]1[OH:7]>[O:11]=[Cr:10](=[O:12])([OH:9])[OH:13]>[CH2:2]1[CH2:1][CH:6]([CH2:5][CH2:4][CH2:3]1)[CH:7]=[O:8]",
        "Jones oxidation",
    ),
    (
        "[CH3:1][CH2:2][CH2:3][CH:4]=[O:5]>[BH4-:6].[Na+:7]>[CH2:4]([OH:5])[CH2:3][CH2:2][CH3:1]",
        "NaBH4 reduction",
    ),
    (
        "[CH2:14]=[CH:13][C:11](=[O:12])[CH3:10].[CH2:8]([O:7][C:5]([CH2:4][C:2](=[O:3])[CH3:1])=[O:6])[CH3:9]"
        ">[Na+].CC[O-]"
        ">[CH3:1][C:2](=[O:3])[CH:4]([C:5](=[O:6])[O:7][CH2:8][CH3:9])[CH2:13]([CH3:14])[C:11](=[O:12])[CH3:10]",
        "Michael addition",
    ),
    (
        "[CH3:12][CH2:11][O:10][C:8](=[O:9])[CH3:7].[CH3:6][CH2:5][O:4][C:2](=[O:3])[CH3:1]>[CH3:16][CH2:15][O-:14].[Na+:13]>[CH2:5]([O:4][C:8]([CH2:7][C:2](=[O:3])[CH3:1])=[O:9])[CH3:6].[CH3:12][CH2:11][OH:10]",
        "Claisen condensation",
    ),
    (
        "[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[Br:7].[cH:9]1[cH:8][c:13]([cH:12][cH:11][cH:10]1)[B:14]([OH:16])[OH:15]>[K+:18].[K+:23].[O-:19][C:20]([O-:22])=[O:21].[Pd:17]>[Br-:7].[OH:15][B:14]=[O:16].[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1-[c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1",
        "Suzuki–Miyaura coupling",
    ),
    (
        "[cH:12]1[cH:11][c:10]([cH:15][cH:14][cH:13]1)[CH:9]=[CH2:8].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[I:7]>[CH2:18]([CH3:17])[N:19]([CH2:20][CH3:21])[CH2:22][CH3:23].[Pd:16]>[I-:7].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)/[CH:8]=[CH:9]/[c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1",
        "Heck reaction",
    ),
    (
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[CH:7]=[O:8].[cH:9]1[cH:10][cH:11][cH:12][cH:13][c:14]1[P+:15]([c:16]1[cH:17][cH:18][cH:19][cH:20][cH:21]1)([c:22]1[cH:27][cH:26][cH:25][cH:24][cH:23]1)[CH2-:28]>[CH2:29]1[CH2:30][CH2:31][O:32][CH2:33]1>[cH:10]1[cH:9][c:28]([cH:13][cH:12][cH:11]1)[P:15]([c:16]1[cH:17][cH:18][cH:19][cH:20][cH:21]1)([c:22]1[cH:27][cH:26][cH:25][cH:24][cH:23]1)=[O:8].[cH:2]1[cH:1][c:6]([cH:5][cH:4][cH:3]1)[CH:7]=[CH2:14]",
        "Wittig olefination",
    ),
]


@pytest.fixture(scope="session")
def sample_reactions() -> list[str]:
    """Return the raw reaction SMILES used by every test."""
    return [r[0] for r in REACTIONS]


@pytest.fixture
def sample_reactions_file(tmp_path: Path, sample_reactions) -> Path:
    """Write the sample reactions to a temporary .smi file."""
    p = tmp_path / "reactions.smi"
    p.write_text("\n".join(sample_reactions))
    return p


# ---------- config factories ---------------------------------------------- #


@pytest.fixture(scope="session")
def std_config() -> ReactionStandardizationConfig:
    """One fully‑loaded standardisation config reused across tests."""
    return ReactionStandardizationConfig(
        reaction_mapping_config=ReactionMappingConfig(),
        functional_groups_config=FunctionalGroupsConfig(),
        kekule_form_config=KekuleFormConfig(),
        check_valence_config=CheckValenceConfig(),
        implicify_hydrogens_config=ImplicifyHydrogensConfig(),
        check_isotopes_config=CheckIsotopesConfig(),
        aromatic_form_config=AromaticFormConfig(),
        mapping_fix_config=MappingFixConfig(),
        unchanged_parts_config=UnchangedPartsConfig(),
        remove_reagents_config=RemoveReagentsConfig(),
        rebalance_reaction_config=RebalanceReactionConfig(),
        duplicate_reaction_config=DuplicateReactionConfig(),
    )


@pytest.fixture(scope="session")
def filt_config() -> ReactionFilterConfig:
    return ReactionFilterConfig(
        cc_sp3_breaking_config=CCsp3BreakingConfig(),
        dynamic_bonds_config=DynamicBondsConfig(min_bonds_number=1, max_bonds_number=2),
    )


@pytest.fixture(scope="session")
def rule_cfg_factory():
    """Return a function to build RuleExtractionConfig variants on demand."""
    base_atom_info = {
        "reaction_center": {
            "neighbors": True,
            "hybridization": True,
            "implicit_hydrogens": False,
            "ring_sizes": False,
        },
        "environment": {
            "neighbors": False,
            "hybridization": False,
            "implicit_hydrogens": False,
            "ring_sizes": False,
        },
    }

    def _factory(**overrides):
        config = {
            "environment_atom_count": 1,
            "multicenter_rules": True,
            "reactor_validation": True,
            "min_popularity": 1,
            "atom_info_retention": base_atom_info,
        }
        config.update(overrides)
        return RuleExtractionConfig(**config)

    return _factory


@pytest.fixture(scope="session")
def simple_molecule():
    """A simple three-atom molecule."""
    return smiles("CCO")


@pytest.fixture(scope="session")
def complex_molecule():
    """A complex molecule with multiple functional groups."""
    return smiles("CC(=O)OC1=CC=CC=C1C(=O)O")


@pytest.fixture(scope="session")
def ring_molecule():
    """A cyclic molecule."""
    return smiles("C1CCCCC1")


@pytest.fixture(scope="session")
def simple_esterification_reaction() -> ReactionContainer:
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH3:6]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH3:6].[OH2:4]"
    )
    # One‑liner gives a fully mapped ReactionContainer
    rxn.canonicalize()  # acts on *all* molecules consistently
    return rxn


@pytest.fixture
def simple_cgr(simple_esterification_reaction) -> CGRContainer:
    """CGR for the simple esterification."""
    # Standard CGR creation - skip if disjoint mapping error occurs
    return ~simple_esterification_reaction


@pytest.fixture
def default_config() -> RuleExtractionConfig:
    """Default RuleExtractionConfig."""
    return RuleExtractionConfig()


@pytest.fixture(scope="session")
def diels_alder_reaction() -> ReactionContainer:
    return smiles(
        "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>"
        "[CH2:1][CH:2]1[CH:3][CH2:4][CH2:5][CH2:6]1"
    )


@pytest.fixture
def diels_alder_cgr(diels_alder_reaction) -> CGRContainer:
    """CGR for the Diels-Alder reaction."""
    # Standard CGR creation - skip if disjoint mapping error occurs
    return ~diels_alder_reaction
