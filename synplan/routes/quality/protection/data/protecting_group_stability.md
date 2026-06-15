# Protecting Group Stability Metadata

`protection_group_templates.csv` includes condition-stability annotations for
protecting groups in these columns:

- `h2o`
- `bases`
- `nucleophiles`
- `electrophiles`
- `reduction`
- `oxidation`

Each cell stores condition labels mapped to numeric stability classes:

- `0`: green, protecting group is stable under these conditions.
- `1`: yellow, protecting group is moderately stable or might react.
- `2`: orange, protecting group is labile.

## Source

The stability information was extracted from the protective groups tables at:

https://www.organic-chemistry.org/protectivegroups/

That website refers to:

T. W. Green, P. G. M. Wuts, *Protective Groups in Organic Synthesis*,
Wiley-Interscience, New York, 1999, 372-381, 383-387, 728-731.
