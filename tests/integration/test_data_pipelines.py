"""Integration tests for the main SynPlanner pipeline components."""

from pathlib import Path

import pytest
from chython import smiles as smiles_chython

from synplan.chem.data.filtering import filter_reactions_from_file
from synplan.chem.data.standardizing import (
    standardize_reactions_from_file,
)
from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions


def debug_standardization(reaction_smiles: str, standardizers: list) -> None:
    """Debug the standardization process for a single reaction.

    Args:
        reaction_smiles: The reaction SMILES string to debug
        standardizers: List of standardizers to apply
    """
    print(f"\nDebugging reaction: {reaction_smiles}")
    try:
        # Parse the reaction
        reaction = smiles_chython(reaction_smiles)
        print(f"Successfully parsed reaction: {reaction}")

        # Apply each standardizer one by one
        current_reaction = reaction
        for i, standardizer in enumerate(standardizers, 1):
            print(f"\nStep {i}: Applying {standardizer.__class__.__name__}")
            print(f"Input reaction: {current_reaction}")
            try:
                current_reaction = standardizer(current_reaction)
                print(f"Output reaction: {current_reaction}")
            except Exception as e:
                print(f"Error in {standardizer.__class__.__name__}: {e!s}")
                print(f"Error type: {type(e)}")
                raise
        print("\nStandardization completed successfully")
    except Exception as e:
        print(f"Failed to process reaction: {e!s}")
        print(f"Error type: {type(e)}")
        raise


# --------------------------------------------------------------------------- #
# 1. Standardisation round‑trip                                               #
# --------------------------------------------------------------------------- #


def test_standardisation_roundtrip(
    tmp_path: Path,
    sample_reactions_file: Path,
    sample_reactions: list[str],
    std_config,
):
    """Test that standardization preserves the number of reactions."""
    out = tmp_path / "std.smi"

    # Create standardizers
    standardizers = std_config.create_standardizers()
    print(f"\nCreated {len(standardizers)} standardizers:")
    for std in standardizers:
        print(f"- {std.__class__.__name__}")

    # Debug the first reaction
    if sample_reactions:
        print("\nDebugging first reaction:")
        debug_standardization(sample_reactions[0], standardizers)

    # Run the full standardization
    standardize_reactions_from_file(
        config=std_config,
        input_reaction_data_path=str(sample_reactions_file),
        standardized_reaction_data_path=str(out),
        num_cpus=1,
    )

    assert out.exists()
    roundtrip = out.read_text().splitlines()
    assert len(roundtrip) == len(sample_reactions)  # Check that no reactions were lost


# --------------------------------------------------------------------------- #
# 2. Filtering keeps at least one reaction                                    #
# --------------------------------------------------------------------------- #


def test_filtering_keeps_some(
    tmp_path: Path,
    sample_reactions_file: Path,
    filt_config,
):
    out = tmp_path / "filt.smi"
    filter_reactions_from_file(
        config=filt_config,
        input_reaction_data_path=str(sample_reactions_file),
        filtered_reaction_data_path=str(out),
        num_cpus=1,
    )

    kept = out.read_text().splitlines()
    assert 0 < len(kept) < len(open(sample_reactions_file).read().splitlines())


@pytest.mark.skip(
    reason="Ray parallel test is slow and has env resolution issues with local chython path"
)
def test_filtering_parallel_matches_serial(
    tmp_path: Path,
    sample_reactions_file: Path,
    filt_config,
):
    serial_out = tmp_path / "filt_serial.smi"
    parallel_out = tmp_path / "filt_parallel.smi"

    filter_reactions_from_file(
        config=filt_config,
        input_reaction_data_path=str(sample_reactions_file),
        filtered_reaction_data_path=str(serial_out),
        num_cpus=1,
        batch_size=2,
    )

    filter_reactions_from_file(
        config=filt_config,
        input_reaction_data_path=str(sample_reactions_file),
        filtered_reaction_data_path=str(parallel_out),
        num_cpus=2,
        batch_size=2,
    )

    serial_lines = serial_out.read_text().splitlines()
    parallel_lines = parallel_out.read_text().splitlines()

    assert serial_lines, "serial filtering removed every reaction"
    assert parallel_lines, "parallel filtering removed every reaction"
    assert serial_lines == parallel_lines


# --------------------------------------------------------------------------- #
# 3. Basic rule extraction returns rules as TSV                               #
# --------------------------------------------------------------------------- #


def test_rule_extraction_basic(
    tmp_path: Path,
    sample_reactions_file: Path,
    rule_cfg_factory,
):
    cfg = rule_cfg_factory()
    out = tmp_path / "rules.tsv"

    extract_rules_from_reactions(
        config=cfg,
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(out),
        num_cpus=1,
        batch_size=2,
    )

    assert out.exists(), "TSV file not created"
    tsv_lines = out.read_text().splitlines()
    assert tsv_lines[0] == "rule_smarts\tpopularity\treaction_indices"
    assert len(tsv_lines) > 1, "no rules extracted"


def test_rule_extraction_tsv_roundtrip(
    tmp_path: Path,
    sample_reactions_file: Path,
    rule_cfg_factory,
):
    """Extract rules → save TSV → load back via load_reaction_rules."""
    from synplan.utils.loading import load_reaction_rules

    cfg = rule_cfg_factory()
    out = tmp_path / "rules.tsv"

    extract_rules_from_reactions(
        config=cfg,
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(out),
        num_cpus=1,
        batch_size=2,
    )

    reactors = load_reaction_rules(str(out))
    assert reactors, "no reactors loaded from TSV"

    # Every loaded reactor should have a valid SMARTS representation
    for reactor in reactors:
        smarts_str = str(reactor)
        assert ">>" in smarts_str, f"invalid SMARTS: {smarts_str}"


# --------------------------------------------------------------------------- #
# 4. Parametrised variants: env atom count & popularity                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("env_cnt", [0, 1, 2])
@pytest.mark.parametrize("popularity", [1, 2])
def test_rule_extraction_variants(
    tmp_path: Path,
    sample_reactions_file: Path,
    rule_cfg_factory,
    env_cnt,
    popularity,
):
    cfg = rule_cfg_factory(environment_atom_count=env_cnt, min_popularity=popularity)
    out = tmp_path / f"rules_env{env_cnt}_pop{popularity}.tsv"

    extract_rules_from_reactions(
        config=cfg,
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(out),
        num_cpus=1,
        batch_size=2,
    )

    assert out.exists(), "TSV file not created"
    tsv_lines = out.read_text().splitlines()
    n_rules = len(tsv_lines) - 1  # subtract header

    # For higher popularity thresholds, we might get no rules
    if popularity == 1:
        assert n_rules > 0  # at least one rule for min_popularity=1
    # stricter popularity => never *more* rules
    if popularity > 1:
        prev_file = tmp_path / f"rules_env{env_cnt}_pop1.tsv"
        if prev_file.exists():
            prev_n = len(prev_file.read_text().splitlines()) - 1
            assert n_rules <= prev_n
