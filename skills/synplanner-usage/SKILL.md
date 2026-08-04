---
name: synplanner-usage
description: >-
  Plan retrosynthetic routes, curate reaction data, extract reaction rules, train
  retrosynthesis policy and value networks, and analyse synthetic routes with the
  installed SynPlanner package. SynPlanner is built on chython (chython-synplan,
  formerly CGRtools), not RDKit. Use when the user imports `synplan.*`, asks for a
  retrosynthesis or synthetic route for a target molecule, trains or tunes a ranking
  or filtering policy or a value network, extracts reaction rules from reaction data,
  standardizes or filters a reaction dataset, clusters or compares routes, or runs
  into chython-versus-RDKit differences such as canonical SMILES that do not match.
  Out of scope: building or contributing to SynPlanner itself, and general
  cheminformatics that does not involve reactions.
license: MIT
compatibility: Requires Python >=3.10,<3.15 and uv. GPU use requires CUDA 12.6 or 12.8.
metadata:
  project: SynPlanner
  homepage: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner
  docs: https://synplanner.readthedocs.io
---

# SynPlanner usage

## What SynPlanner is

SynPlanner is a high-level reaction processing toolbox for retrosynthesis. It
orchestrates **chython-synplan** (also known as chython, formerly CGRtools) — a
pure-Python chemoinformatics engine with capabilities comparable to RDKit — for
reaction data curation, retrosynthetic planning, and analysis of synthetic routes.

## When to use something else

SynPlanner is for retrosynthesis and reaction-level work. It does not need to be in
the loop for plain molecule handling.

chython is already installed as SynPlanner's engine and covers most of what RDKit
does. For molecule-level work that never touches a reaction or a route, use
`from chython import ...` directly rather than going through SynPlanner. Do not
reach for RDKit just because the task looks like ordinary cheminformatics — chython
is already there.

Use RDKit only where chython genuinely lacks the capability. SynPlanner itself does
exactly this for synthetic-accessibility scoring and molecular descriptors — see
`synplan/chem/rdkit_utils.py`.

Do not overcorrect. "Prefer chython" is not "never RDKit" — when chython does not
have the function, reach for RDKit and say so, rather than hand-rolling it.

## Installing

Python `>=3.10,<3.15`. Three supported paths:

```bash
pip install SynPlanner                 # users — recommended, use a virtualenv
uv sync --extra cpu                    # from a source clone (dev)
docker build --platform linux/amd64 -t synplan:cli -f cli.Dockerfile .
```

Docker is the most reproducible and the fallback when a platform gives trouble.
Check the result with `synplan --version`.

## Two things the install does not give you

**1. Data and model weights are not bundled.**

```bash
synplan download_preset --preset synplanner-gps --save_to synplan_data
```

Fetches from HuggingFace and prints a `key: path` map. Those paths are what
`synplan planning` consumes. `download_all_data` still exists but is
**deprecated** — do not use it.

**2. `configs/*.yaml` are not installed by pip.** They live only in the git
repository. Either clone it, or fetch the one you need:

```bash
curl -O https://raw.githubusercontent.com/Laboratoire-de-Chemoinformatique/SynPlanner/main/configs/planning_standard.yaml
```

`synplan planning --config` is required and has no built-in default.

## When a GPU matters

Only two workflows benefit from a GPU: **reaction mapping** (`reaction_mapping`)
and **network training** (`ranking_policy_training`, `filtering_policy_training`,
`mhn_network_tuning`, `value_network_tuning`). Everything else — planning, data
curation, rule extraction, route analysis — runs on CPU, which is the typical case.

If the user needs one of those two on a GPU, set it up for them: read
[Selecting a PyTorch build](https://synplanner.readthedocs.io/en/latest/get_started/installation.html#selecting-a-pytorch-build)
and run the install yourself. Do not hand the user a link and stop.

## chython is not RDKit

Default to SynPlanner-native output.

Ask which the user wants **up front** only when one of these is true:

- their existing code imports `rdkit`
- results will be compared against external data
- the output feeds another tool

Otherwise assume native and do not interrupt. When every step is complete, ask once
whether to verify RDKit compatibility of the final result.

**Never compare SMILES strings across engines.** Canonical SMILES from chython
differ from canonical SMILES from RDKit for the same molecule. To compare
molecules, either parse both through the same engine and compare the resulting
objects, or compare InChI.

When the user needs RDKit objects at the boundary, use `synplan.chem.rdkit_compat`
rather than converting by hand:

| Need | Function |
| --- | --- |
| Pass an RDKit `Mol` as the target | `target_from_rdkit(...)` |
| Pass RDKit `Mol`s as building blocks | `building_blocks_from_rdkit(mols)` |
| Get one route back as RDKit `Mol`s | `route_to_rdkit(tree, node_id)` |
| Get all routes back as RDKit `Mol`s | `extract_routes_rdkit(tree)` |

Requires `rdkit` to be installed.

## Write the least code possible

Before writing code, check whether SynPlanner already does the task. Stop at the
first rung that holds:

1. **A CLI command.** Run `synplan --help`. The CLI covers the full pipeline —
   see the table below.
2. **A documented module.** Anything under `docs/api_reference/`. Note that
   `synplan.Tree` is the only symbol exported at top level.
3. **Other functions inside SynPlanner.**
4. **chython directly.** Last resort, only when SynPlanner does not cover it.

The ladder shortens the code, never the checking. Reimplementing something that
already exists is the common failure here — not writing too much code. Look before
you write.

The reverse failure is as bad: do not force a rung that does not fit. Three
SynPlanner calls chained into a workaround is worse than one honest chython call.

### CLI commands

| Stage | Commands |
| --- | --- |
| Data | `download_preset` |
| Preparation | `building_blocks_standardizing`, `ord_convert` |
| Curation (in order) | `reaction_mapping` → `reaction_standardizing` → `reaction_filtering` |
| Rules | `rule_extracting` |
| Training | `ranking_policy_training`, `filtering_policy_training`, `mhn_network_tuning`, `value_network_tuning` |
| Planning | `planning` |
| Analysis | `clustering` |

Prefer the CLI over Python for long-running work.

### Which API pieces a task needs

For the call sequence behind any task — planning, clustering, route comparison,
data curation, rule extraction, training — read `references/tasks.md`, next to
this file, or the
[published copy](https://synplanner.readthedocs.io/en/latest/tasks.html). It maps
each task to the exact functions and their order, and links to the worked
notebook and reference page.

## Use a config from `configs/`, do not write YAML from scratch

Eleven task configs live in the repository (not installed by pip — see above).
Copy the closest one and edit it:

| Task | Config |
| --- | --- |
| Planning | `planning_standard.yaml`, `planning_value.yaml`, `planning_combined_policies.yaml` |
| Rule extraction | `rules_extraction.yaml`, `extraction_functional_groups.yaml` |
| Data curation | `reactions_standardization.yaml`, `reactions_filtration.yaml` |
| Policy training | `policy_training.yaml`, `mhn_ranking_policy_training.yaml`, `combined_ranking_filtering_policy.yaml` |
| Tuning | `tuning.yaml` |

They are small. `planning_standard.yaml` has two sections, `tree` and
`node_expansion`. Read the config before changing any value.

## Running a plan, and what comes back

```bash
synplan planning \
  --config configs/planning_standard.yaml \
  --targets <targets-file> \
  --reaction_rules <path> \
  --building_blocks <path> \
  --policy_network <path> \
  --results_dir ./results
```

Four data paths are required and all come from `download_preset`.
`--value_network` is optional.

Two flags change the output:

- `--export_routes` also writes `results.json.gz` (target-keyed routes) and
  `manifest.json` for downstream consumers.
- `--reconcile-mapping` reconciles atom-map numbering across steps. **Roughly 4x
  slower.** The default uses per-step-local atom numbering.

## Before final delivery

When every step is complete — not before — ask the user whether to verify RDKit
compatibility of the final result.

## Going deeper

- Documentation: <https://synplanner.readthedocs.io>
- Tutorials, in order: `docs/user_guide/` — start with `00_Welcome_to_Chython`
  and `01_Coming_from_RDKit`, then `ten_minutes`
- RDKit interop worked example: `11_Planning_with_RDKit`
