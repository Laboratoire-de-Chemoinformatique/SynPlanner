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

The unmodified cyanamide/methanol reaction is passed to `extract_rules` with
`as_query_container=False`. With validation enabled or disabled, the reverse rule
must retain the recorded product and both starting materials. The enabled case
also requires `reactor_validation == "passed"`.

## Stereo-valid match retention

The nitrile reduction tests derive reverse templates from the mapped source:

1. Use the recorded product string verbatim as the pattern, preserving both
   stereo annotations and atom order.
2. Retain the source reactant component that shares atom maps with the product.
   This removes only ammonia/water and excludes the reagent-side catalyst.
3. Use the substrate verbatim as the full RHS, or retain only its nitrile atoms
   (maps 4 and 14) for a partial RHS. A nitrogen-only RHS (map 14) leaves the
   target unchanged and covers the single-shared-atom case.
4. Load each template through the public TSV loader with default filtering and
   with `automorphism_filter=False`. Both must return the recorded substrate
   (or unchanged target for the nitrogen-only patch), after removing stereo to
   match `CanonicalRetroReactor`'s flat-output contract.

These are templates derived from raw USPTO data, not entries from the bundled
GPS/GCN rule TSVs. The symmetric ring paths expose loss of the only stereo-valid
match, including when the permuted atoms are absent from the RHS.

## Running

From `SynPlanner`:

```sh
uv run pytest -o addopts='' -q tests/regression/test_symmetric_rule_real_data.py
```
