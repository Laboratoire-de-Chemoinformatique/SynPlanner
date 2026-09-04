# USPTO fixtures for PR #104

`pr104_uspto.json` contains two verbatim source rows, including their record IDs
and patent identifiers. Source paths are relative to the local `Dev/synplan`
workspace. Line numbers are one-based. Each SHA-256 covers the UTF-8 source row
without its trailing newline. The tests use the checked-in copies, so neither
the large source datasets nor network access is required.

| Fixture | Local source | Line | Record | Patent |
| --- | --- | ---: | ---: | --- |
| `cyanamide_methanol` | `data_repo/SynPlanner/uspto/uspto_standardized.zip`, member `uspto_standardized.smi` | 552 | 581 | US03931316 |
| `stereo_nitrile_reduction` | `SynPlanner/local/uspto_full_mapped.smi` | 4509 | 4509 | US03950405 |

## Molecule-container extraction

The first reaction adds methanol to cyanamide, producing methyl carbamimidate.
All five heavy atoms are mapped and retained. The test passes the source reaction
directly to `extract_rules` with the documented `as_query_container=False` option.
No reaction edits or template substitutions are made.

The control disables reactor validation and checks that the reverse rule retains
the exact source product and both starting materials. The regression enables
validation and additionally requires `reactor_validation == "passed"`.
At PR commit `8f0b80c`, validation instead raises
`TypeError: 'int' object is not iterable` inside `_query_atoms_overlap`.
Both extraction cases pass against the code archived from `origin/main` commit
`0598e7a9adfaf1d642934f09ceb29ffd9cd60a1d`, using the same Python environment.

## Stereo-valid match retention

The second reaction reduces a nitrile substituent on a stereochemically specified
1,4-disubstituted cyclohexane to an aminomethyl group. Its input row contains the
nitrile substrate, two ammonia molecules, water, and a cobalt catalyst.

The test derives a full-substrate reverse SMARTS directly from the mapped source:

1. Use the recorded product string verbatim as the pattern, preserving both
   stereo annotations and atom order.
2. Retain the source reactant component that shares atom maps with the product.
   This removes only ammonia/water and excludes the reagent-side catalyst.
3. Use that substrate string verbatim as the replacement template. No atoms,
   bonds, maps, substituents, or stereo annotations are added or rewritten.
4. Load the template through the public TSV loader and apply it to the recorded
   product. Compare the result with the recorded nitrile substrate after removing
   stereo, matching `CanonicalRetroReactor`'s flat-output contract.

At `8f0b80c`, the explicit `automorphism_filter=False` control produces the
recorded nitrile precursor; default loading produces no reaction. The symmetry
detector accepts an exchange of the cyclohexane paths as RHS-preserving, but
Chython can keep a stereo-invalid match and discard the valid one before stereo
validation. The test therefore exercises the same missing-match issue as the
review's constructed example using an actual reaction and unchanged structures.

This is a template derived from raw USPTO data, not a claim that a stereo-bearing
template exists in the bundled GPS/GCN rule TSVs. Those files contain no stereo
annotations. It is a remaining gap in the PR's new detection policy; the
molecule-container crash is a regression relative to `main`.

## Running

From `SynPlanner`:

```sh
uv run pytest -o addopts='' -q tests/regression/test_symmetric_rule_real_data.py
```

The expected result on the reviewed commit is **two passed controls and two
failed regression contracts**. The failures are deliberately not marked xfail:
they should turn green when the defects are fixed.

Together with the existing rule-loading, symmetry-detection, extraction-audit,
and extraction-unit suites, validation produced **78 passed and these two
failures**. Ruff lint and formatting checks passed.
