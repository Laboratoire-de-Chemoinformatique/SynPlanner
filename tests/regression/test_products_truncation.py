"""A capped enumeration is a depth-first prefix, so it must say so."""

import warnings

import pytest

from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.synthon.enumerate import Enumerator, ProductsTruncatedWarning
from synplan.chem.synthon.stock import load_synthon_stock
from synplan.interfaces.synthon_commands import synthonise_file

WIDE = {"mw_lower": 0.0, "mw_upper": 10_000.0}


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthons") / "stock.smi"
    synthonise_file("tests/data/synthon/BBs.cxsmiles", str(out), SynthonConfig())
    return sorted(load_synthon_stock(str(out)))


def test_capped_enumeration_warns_that_it_truncated(stock):
    config = SynthonConfig(max_products=5, **WIDE)
    with pytest.warns(ProductsTruncatedWarning, match="depth-first"):
        products = list(Enumerator(config).enumerate_library(stock))
    assert len(products) == 5


def test_uncapped_enumeration_is_silent(stock):
    config = SynthonConfig(max_products=100_000, **WIDE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        products = list(Enumerator(config).enumerate_library(stock))
    assert len(products) > 5, (
        "fixture must exhaust below the cap or this proves nothing"
    )
    assert not [w for w in caught if issubclass(w.category, ProductsTruncatedWarning)]
