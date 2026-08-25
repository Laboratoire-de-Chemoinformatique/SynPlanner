from synplan.chem.reaction.curation.config import SmallMoleculesConfig
from synplan.chem.reaction.curation.filtering import (
    SmallMoleculesConfig as FilteringSmallMoleculesConfig,
)
from synplan.chem.reaction.curation.standardizing import (
    SmallMoleculesConfig as StandardizingSmallMoleculesConfig,
)


def test_small_molecules_config_has_one_shared_definition():
    assert FilteringSmallMoleculesConfig is SmallMoleculesConfig
    assert StandardizingSmallMoleculesConfig is SmallMoleculesConfig

    config = SmallMoleculesConfig.model_validate({"mol_max_size": 8})
    assert config.mol_max_size == 8
