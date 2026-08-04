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

`synplan --help` lists every command. The CLI covers the whole pipeline — ORD
import, curation, rule extraction, training, planning, clustering — so check it
before writing Python, and prefer it for long-running work.

## Start every task from `references/tasks.md`

That file sits next to this one — also published
[here](https://synplanner.readthedocs.io/en/latest/tasks.html) — and maps around
thirty tasks to the exact API pieces each needs, in order, with links to a
worked notebook and the reference page.

**This file gives the rules. That one gives the sequences.** No call sequence is
repeated here, so working without it means reconstructing one from the source,
which is where the time goes.

How to use it:

1. Scan the bold titles for the one matching the request. They are phrased the
   way users ask — "find a synthesis route for a molecule", "clean a reaction
   dataset", "compare two sets of routes", "nothing was found".
2. Read that entry. It names the functions and the order to call them in.
3. Follow its links only if you need running code or the reasoning behind a
   parameter.
4. Check the combinations section first — most real requests are a chain of two
   or three entries, not one.

Read it before writing planning, route analysis, comparison, clustering,
curation or training code.

**Do not write your own route-tree walker.** Recursing over the route structure
to count depth or collect precursors is the most common reinvention here, and
every one of those traversals already exists — `extract_routes`,
`export_tree_to_json`, `compose_all_route_cgrs` and the rest are listed in
`tasks.md` under "Working with routes".

## Use a config from `configs/`, do not write YAML from scratch

One shipped config per task lives in `configs/` in the repository — `ls` it.
Copy the closest and edit it; they are small, and reading one takes less time
than getting a hand-written config subtly wrong.

### Start with the defaults, then adjust when the data says to

The shipped configs are tuned for large public datasets. On a smaller or
different corpus a stage can legitimately return nothing — standardization
rejecting every reaction, rule extraction writing a file with only a header,
planning finding no routes. None of these raise an error.

**An empty result is a signal to tune, not a result to report.** When a stage
returns nothing:

1. Read the matching page under `configuration/` and `methods/` in the docs.
   Every parameter is documented there with its default and what it filters.
2. Find the parameter that is excluding the data — popularity and frequency
   thresholds, reagent handling, size and validity filters are the usual ones.
3. Change it deliberately, re-run that stage, and check the count moved.
4. Look at what else is available before concluding it cannot work — other
   configs, other strategies, other evaluation modes. The default path is not
   the only path.

Say which parameter you changed and why. Stopping at "the pipeline produced
nothing" when a threshold tuned for a million reactions was applied to a hundred
is not a finding.

**Check the chemistry before you tune anything.** Rules extracted from a dataset
can only disconnect bonds that dataset contains. A corpus of C–N couplings cannot
produce a route to a molecule with no C–N bond, at any threshold. Before treating
an empty result as a tuning problem, look at what reactions are actually in the
set and whether they could reach the target at all. If they cannot, say so and
stop — lowering thresholds against chemistry that is not there wastes time and
ends in a route that does not exist. Tuning fixes a filter that is too strict; it
does not fix a corpus that is the wrong corpus.

## Two defaults that are easy to miss

**`tree.run()` runs the search.** A `Tree` does no work when constructed — build
one, call `run()`, then read its results. `run()` returns the tree, so it chains.
Iterate the tree instead only when you need the per-iteration
`(is_solved, node_ids)` for progress or early stopping. Older code exhausts the
iterator with `list(tree)`; that still works but allocates results only to
discard them.

**Pass `route_scorer=ProtectionRouteScorer.from_config()` to `Tree` by
default.** Someone asking for a synthesis wants routes they can act on, and the
raw search output is not ranked by quality — unranked routes are the common
disappointment. Omit it only when the user explicitly asks for the unfiltered
tree. There is no CLI flag for it; this is Python-only.

## Before final delivery

When every step is complete — not before — ask the user whether to verify RDKit
compatibility of the final result.

## Going deeper

- Documentation: <https://synplanner.readthedocs.io>
- Tutorials, in order: `docs/user_guide/` — start with `00_Welcome_to_Chython`
  and `01_Coming_from_RDKit`, then `ten_minutes`
- RDKit interop worked example: `11_Planning_with_RDKit`
