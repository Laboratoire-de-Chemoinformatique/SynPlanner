# Stereo inheritance fixtures

`routes.json` contains 11 unchanged source/planner route trees selected from the
prospectively frozen Agent 2 development panel. The full panel is stored at
`benchmarks/audit-2026-09-05/strategy/agent_2/panel.json` in the surrounding
workspace. Source routes come from the already designated mkt-cnv-160 development
set; reaction metadata retain patent and USPTO reaction record identifiers.
Planner routes are saved GPS/PUCT proposals, not new search runs.

- Source definitions SHA-256: `7390a9c2522b1c81605e80a42a8de7c7ad9a5797d028633b54904bc78f091bb8`.
- Saved planner pool SHA-256: `a32164fac3ff27f8552d41801d5a43471a5f1598e9de1bf694832b19f4073126`.
- `stock.json` retains all 178 original catalogue rows in the 102 leaf
  connectivity buckets requested by the full panel. It includes source CSV line
  numbers (header is line 1), SMILES and full keys. The complete 313,458-record
  `buyables-stock.csv.gz` SHA-256 is
  `aee42d44c9a96279dd13732d8a298bcfcad34a3ac0ba6b4b2715ec36ad18ec0b`.

`cip_change.json` contains a separately sourced **single-step regression**, USPTO
record 561, with the exact original row, patent identifiers and row hash. Atom 12
retains its mapped orientation while changing CIP S to R. This is not an extra
route in the coverage panel; the same source reaction also forms another centre.
Use RDKit `rdCIPLabeler.AssignCIPLabels` for CIP checks: the legacy labels on mapped
molecules can use atom maps as a ranking input.

The negative stock, corrupted intermediate, conflict, price, and loss/recreation
tests are explicitly controlled modifications of these real substrates/routes.
They are not additional literature claims. Price values in the price test are
synthetic; the real catalogue snapshot contains no prices. The natural wrong-stock
case is `planner:n5-00611`; the natural symmetry case is `source:n5-08737`.

No URSA/ChemCensor scores, outputs or restricted evaluation loop were used.
