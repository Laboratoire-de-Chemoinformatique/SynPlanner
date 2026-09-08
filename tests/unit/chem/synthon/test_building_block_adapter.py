from unittest.mock import Mock

import pytest
from frozendict import frozendict

from synplan.chem.building_blocks import BuildingBlock
from synplan.chem.synthon.synthonise import BBSynthoniser


def test_building_block_adapter_delegates_only_the_smiles():
    synthoniser = object.__new__(BBSynthoniser)
    expected = {"[CH3_elec]": {"classes": {"test"}, "component": 0}}
    synthoniser.synthonise_smiles = Mock(return_value=expected)
    block = BuildingBlock(
        smiles="CCO",
        inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        vendors=frozendict({"vendor": 1.0}),
        has_stereo=False,
    )

    assert synthoniser.synthonise_building_block(block) is expected
    synthoniser.synthonise_smiles.assert_called_once_with("CCO")


def test_building_block_adapter_rejects_other_records():
    synthoniser = object.__new__(BBSynthoniser)
    with pytest.raises(TypeError, match="BuildingBlock"):
        synthoniser.synthonise_building_block("CCO")
