"""PR #104 contracts using verbatim, mapped USPTO records from local datasets.

See tests/data/regression/pr104_uspto.md for provenance and template derivation.
The validation-enabled and default-filter cases cover previously lost mappings;
their paired controls verify that the source chemistry is usable.
"""

import json
from pathlib import Path

import pytest
from chython import smarts, smiles

from synplan.chem.reaction.config import ReactorConfig
from synplan.chem.reaction.rules.config import RuleExtractionConfig
from synplan.chem.reaction.rules.extraction import extract_rules
from synplan.utils.loading import load_reaction_rules


@pytest.fixture(scope="module")
def uspto_records():
    path = Path(__file__).parents[1] / "data" / "regression" / "pr104_uspto.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "reactor_validation", [False, True], ids=["control", "validate"]
)
def test_molecule_rule_extraction_from_uspto_cyanamide_methanol(
    uspto_records, reactor_validation
):
    """US03931316: cyanamide + methanol -> methyl carbamimidate.

    This record validates with main's validate_rule implementation. Passing a
    MoleculeContainer to the new query-only symmetry analysis must not crash.
    """
    reaction_smiles = uspto_records["cyanamide_methanol"]["source_line"].split("\t")[0]
    source = smiles(reaction_smiles)
    config = RuleExtractionConfig(
        as_query_container=False,
        reactor_validation=reactor_validation,
        min_popularity=1,
    )

    rules, skipped = extract_rules(config, source)

    assert not skipped
    assert len(rules) == 1
    (rule,) = rules
    # All five mapped atoms participate: extraction must retain the recorded
    # product and both recorded starting materials in the reverse direction.
    assert set(rule.reactants) == set(source.products)
    assert set(rule.products) == set(source.reactants)
    assert rule.meta.get("reactor_validation") == (
        "passed" if reactor_validation else None
    )


@pytest.mark.parametrize("disable_filter", [True, False], ids=["control", "default"])
@pytest.mark.parametrize(
    "rhs_atoms",
    [None, (4, 14), (14,)],
    ids=["full-rhs", "partial-rhs", "single-atom-rhs"],
)
def test_stereo_nitrile_reduction_keeps_recorded_precursor(
    uspto_records, tmp_path, disable_filter, rhs_atoms
):
    """US03950405: the reverse of a recorded nitrile-to-amine reduction.

    Keep the mapped product verbatim as SMARTS. The RHS is the contributing
    substrate, its nitrile group, or just its nitrogen (an unchanged patch).
    Symmetric ring paths allow Chython's filter to discard the only
    stereo-valid match even though the RHS is invariant under the path exchange.
    """
    source_smiles = uspto_records["stereo_nitrile_reduction"]["source_line"].split(
        "\t"
    )[0]
    reactants, _reagents, product = source_smiles.split(">")
    target = smiles(product)
    product_maps = set(target.atoms_numbers)
    contributing_substrates = [
        component
        for component in reactants.split(".")
        if product_maps.intersection(smiles(component).atoms_numbers)
    ]
    assert len(contributing_substrates) == 1
    (substrate,) = contributing_substrates
    assert sum(atom.stereo is not None for _, atom in target.atoms()) == 2
    replacement = substrate
    if rhs_atoms is not None:
        patch = smarts(substrate)
        for number in set(patch.atoms_numbers) - set(rhs_atoms):
            patch.delete_atom(number)
        replacement = format(patch, "m")
    rule_smarts = f"{product}>>{replacement}"
    rules_path = tmp_path / "nitrile_reduction.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{rule_smarts}\t1\t0\n",
        encoding="utf-8",
    )
    kwargs = (
        {"reactor_config": ReactorConfig(automorphism_filter=False)}
        if disable_filter
        else {}
    )
    try:
        (reactor,) = load_reaction_rules(str(rules_path), **kwargs)
        precursor_sets = {
            tuple(sorted(str(molecule) for molecule in result.products))
            for result in reactor(target)
        }
    finally:
        load_reaction_rules.cache_clear()

    # The nitrogen-only patch leaves the target unchanged. Compare with the
    # recorded molecules using CanonicalRetroReactor's flat-output contract.
    expected = smiles(product if rhs_atoms == (14,) else substrate)
    expected.clean_stereo()
    assert precursor_sets == {(str(expected),)}
