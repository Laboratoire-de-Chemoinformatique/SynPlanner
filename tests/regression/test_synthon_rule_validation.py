"""No shipped ring rule may label an atom whose proton canonicalisation moves."""

import pytest
from chython import smiles, synthon_smiles

from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.rules.validate import (
    labelled_atoms_survive_canonicalisation,
    shifted_labels,
)
from synplan.chem.synthon.transformer import SynthonTransformer

from .test_synthon_enumeration import RING_EXAMPLES

# every triad that killed a rule during the heterocyclisation port, with the reaction it cost
TAUTOMER_TRAPS = {
    "imidic acid (oxazole)": "c1ccccc1C([OH_nuc])=[NH_nuc]",
    "thioamide (thiazole)": "c1ccccc1C([SH_nuc])=[NH_nuc]",
    "amidine (pyrimidine)": "c1ccccc1C(=[NH_nuc])[NH2_nuc2]",
    "enol": "C[CH_nuc]=[C_nuc](C)[OH]",
}


def _ring_records():
    """The R16 families, which have a hand-authored target here.

    The R17 block is checked the same way against its own authored targets by the converter's
    `check()`, which `test_synthon_data_is_current` runs.
    """
    data = load_data(SynthonConfig().rules_path)
    return [r for r in data["disconnections"] if r["id"] in RING_EXAMPLES]


@pytest.mark.parametrize("record", _ring_records(), ids=lambda r: r["id"])
def test_a_shipped_ring_rule_keeps_the_regiochemistry_it_spells(record):
    """The rule set is only trustworthy while the written fragment is the one that comes out."""
    target = smiles(RING_EXAMPLES[record["id"]])
    target.canonicalize()
    rule = SynthonTransformer.from_smarts(record["smarts"])
    for synthon in next(iter(rule(target))).split():
        assert labelled_atoms_survive_canonicalisation(synthon), (
            f"{record['id']}: {shifted_labels(synthon)}"
        )


@pytest.mark.parametrize("name,smi", TAUTOMER_TRAPS.items())
def test_the_guard_catches_a_known_trap(name, smi):
    """Without this the guard could pass by never firing, which is how a guard rots."""
    assert not labelled_atoms_survive_canonicalisation(synthon_smiles(smi)), name


def test_a_blocked_triad_is_not_flagged():
    """R16.4's N-methyl stops the shift, so an amidine is not automatically disqualified."""
    assert labelled_atoms_survive_canonicalisation(
        synthon_smiles("c1cc(ccc1)C([NH_nuc]C)=[NH_nuc]")
    )
