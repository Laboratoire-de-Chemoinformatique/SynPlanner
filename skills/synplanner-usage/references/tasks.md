# SynPlanner task index

Scan the bold titles, pick the ones that match the request, then open the linked
pages. Entries give the API pieces and their order — not working code. The
tutorials have the code; the docs have the reasoning and the knobs.

**Doc links** are relative to <https://synplanner.readthedocs.io/en/latest/> —
`methods/planning` means `.../methods/planning.html`. In a clone they are
`docs/methods/planning.rst`. **Tutorial links** are notebooks in
`docs/user_guide/`.

## Planning setup

Most route-level tasks open with the same block. Entries below say "planning
setup, then ..." rather than repeat it.

| Step | Where |
| --- | --- |
| `download_preset` | `synplan.utils.loading` |
| `load_building_blocks`, `load_reaction_rules`, `load_policy_function` | `synplan.utils.loading` |
| `TreeConfig`, `RolloutEvaluationConfig` | `synplan.utils.config` |
| `load_evaluation_function` | `synplan.utils.loading` |
| `mol_from_smiles` | `synplan.chem.utils` |
| `ProtectionRouteScorer.from_config()` | `synplan.chem.reaction.routes.quality.scorer` |
| `Tree` | `synplan.mcts.tree` |

`download_preset` returns a dict with keys `building_blocks`, `reaction_rules`,
`ranking_policy`. **`tree.run()` runs the search** and returns the tree; a `Tree`
does nothing when constructed. Iterate it instead only for per-iteration
`(is_solved, node_ids)` — progress, or early stopping.
Docs: `methods/mcts` — "Running a search"

Pass the scorer as `Tree(..., route_scorer=route_scorer)` so routes come back
re-ranked by quality. **Do this by default.** Users asking for a synthesis want
routes they can act on, not raw search output; unranked routes are the common
disappointment. Drop the scorer only when the user explicitly wants the
unfiltered tree.

## Common combinations

Most real requests are a chain, not a single entry.

**"Give me good routes for this molecule"** — the default ask
Planning setup *with* `route_scorer` → `tree.run()` → `extract_routes` /
`get_route_svg`. This, not bare planning, is the baseline answer.

**"Give me good routes, and show me how they differ"**
The above, then `export_tree_to_json` and `routes_clustering_report`.

**"Clean my reaction data"**
`map_reactions_from_file` once, then `standardize_reactions_from_file` and
`filter_reactions_from_file` — those two effectively always run as a pair.

**"Use my own reaction data end to end"**
Clean → `extract_rules_from_reactions` → `create_policy_dataset` +
`run_policy_training` → planning setup with the new rules and policy.

Check the rule count before training. Policy training needs a corpus large
enough to hold out a validation split; a handful of rules will not train and the
run fails on the missing validation metric. If extraction yields only a few
rules - **use priority rules instead of
training**, see "Plan with my own retrosynthetic SMARTS" below.

## Finding routes

**Find a synthesis route for a molecule**
Planning setup with `route_scorer`, then `extract_routes` / `get_route_svg`
(`synplan.utils.visualisation`).
Tutorial: `05_Retrosynthetic_Planning`, `ten_minutes`
Docs: `methods/planning`, `methods/mcts`, `configuration/planning`

**Plan many molecules at once**
`run_search` (`synplan.mcts.search`) with `PolicyNetworkConfig` and
`RolloutEvaluationConfig`. Targets are a `.smi` file.
CLI: `synplan planning --config --targets --reaction_rules --building_blocks
--policy_network --results_dir`. Those four data paths are required and all come
from `download_preset`; `--value_network` is optional. `--export_routes` also
writes `results.json.gz` (target-keyed) and `manifest.json`;
`--reconcile-mapping` reconciles atom-map numbering across steps but is roughly
4x slower. Prefer the CLI for large batches.
Docs: `user_guide/cli_interface`, `configuration/planning`

**Tune the search — depth, time, breadth, quality**
`TreeConfig` fields: `max_iterations`, `max_time`, `max_depth`, `max_tree_size`,
`search_strategy`, `ucb_type`, `c_ucb`. `node_expansion` controls `top_rules` and
`rule_prob_threshold`.
Docs: `configuration/planning`, `methods/mcts`

**Nothing was found / routes look wrong**
Check building blocks first — a target is only solved when it reaches them. Then
widen `max_iterations` / `max_depth`, or loosen `top_rules`.
Docs: `configuration/planning`, `methods/mcts`, `user_guide/data`

**Try a different search algorithm**
`TreeConfig.search_strategy`; NMCS and other variants are documented under
"Alternative Search Algorithms".
Tutorial: `10_NMCS_Algorithms`
Docs: `methods/mcts`

**Plan with RDKit molecules in and out**
Planning setup, but swap `mol_from_smiles` → `target_from_rdkit` and
`load_building_blocks` → `building_blocks_from_rdkit`. Results via
`route_to_rdkit` / `extract_routes_rdkit`. All `synplan.chem.rdkit_compat`.
Tutorial: `11_Planning_with_RDKit`

**Plan with my own retrosynthetic SMARTS**
Also the answer when a custom dataset is too small to train a policy on — hand
the chemistry over as priority rules rather than trying to learn it from a few
examples. Combine priority rules with the preset policy.
`parse_priority_rules` (`synplan.chem.reaction.rules`) →
`Tree(config=TreeConfig(use_priority=True), priority_rules=...)`.
**Python API only — there is no CLI flag for priority rules**, so this is one of
the few tasks where `synplan planning` is not the answer.
Tutorial: `13_Priority_Rules`
Docs: `methods/priority_rules` — read the SMARTS dialect note before writing any

**Combine ranking and filtering policies**
`load_combined_policy_function` (`synplan.utils.loading`) applied to a
`Precursor` (`synplan.chem.precursor`).
Tutorial: `09_Combined_Ranking_Filtering_Policy`
Docs: `methods/policy`, `configuration/planning`

**Guide search with a value network**
`load_evaluation_function` with a value-network config instead of rollout.
CLI: `value_network_tuning` to train one.
Docs: `methods/value`, `configuration/value`

## Working with routes

**See and export routes**
`extract_routes`, `get_route_svg`, `generate_results_html`
(`synplan.utils.visualisation`); `export_tree_to_json`, `export_tree_to_csv`,
`make_json`, `read_routes_json` (`synplan.chem.reaction.routes.io`).
Tutorial: `07_Clustering`
Docs: `methods/routes` — see "Typed Route APIs"

**Rank routes by quality / avoid protecting-group problems**
Already in the planning setup: `ProtectionRouteScorer.from_config()` passed as
`Tree(route_scorer=...)`. Go lower level only to tune it — `ProtectionConfig`,
`FunctionalGroupDetector`, `get_reaction_center_atoms`, `classify_reaction_type`.
Tutorial: `08_Protection_Scoring`

**Group similar routes together**
Planning setup, then export, then `cgr_display`
(`...routes.representation.depiction`) and `routes_clustering_report`.
CLI: `synplan clustering`.
Tutorial: `07_Clustering`
Docs: `methods/routes`

**Compare two sets of routes**
`compose_all_route_cgrs`, `compare_route_cgr_dicts`, `route_cgr_overlap_rows`
(`...routes.representation`); plot with `plot_sb_cgr_cluster_venn`
(`...routes.notebook_plots`).
Tutorial: `15_Routes_compare`
Docs: `methods/routes` — RouteCGR deconvolution

**Check which building blocks a route uses**
`compose_all_route_cgrs`, `route_ids_with_exact_bb`, `collect_bb_usage_stats`.
Tutorial: `16_Building_block_search`

**Analyse the search itself — policy performance, branching, depth**
No SynPlanner API involved: reads exported JSON/CSV and plots with matplotlib.
`Tree.stats` is a typed dataclass.
Tutorial: `06_Tree_Analysis`
Docs: `methods/mcts` — "Tree Analytics"

## Data and rules

**Get the data and models**
`download_preset` for a ready bundle; `download_unpack_data` for datasets.
CLI: `synplan download_preset`.
Docs: `get_started/data_download`, `user_guide/data`

**Clean a reaction dataset**
`MappingConfig` + `map_reactions_from_file` (`synplan.chem.data.mapping`), then
`standardize_reactions_from_file`, then `filter_reactions_from_file`.
CLI: `reaction_mapping` → `reaction_standardizing` → `reaction_filtering`.
Mapping is the one stage where hardware matters: CPU is fine to roughly ten
thousand reactions, beyond that use `MappingConfig(device=...)` on a GPU.
chython's `reset_mapping()` maps a single reaction in memory and is the wrong
tool for a file.
Tutorial: `02_Data_Curation`
Docs: `methods/standardization` — "Standardization order" is authoritative, and
"Two ways to map" covers the hardware choice — `methods/filtration`,
`configuration/standardization`, `configuration/filtration`

**Extract reaction rules from my own reactions**
`RuleExtractionConfig` (`synplan.utils.config`) → `extract_rules_from_reactions`
(`synplan.chem.reaction.rules.extraction`) → `load_reaction_rules` to check.
CLI: `rule_extracting`.
Tutorial: `03_Rules_Extraction`
Docs: `methods/extraction`, `configuration/extraction`

**Inspect or visualise a rule set**
`RuleSet` (`synplan.chem.reaction.rules.analysis`), `query_to_mol`
(`...rules.representation`).
Tutorial: `12_Rule_Analysis`
Docs: `methods/extraction`

**Use my own building blocks**
CLI: `building_blocks_standardizing`. Preset building blocks are already
standardized — pass `standardize=False` to `load_building_blocks`.
Docs: `user_guide/cli_interface`, `user_guide/data`

**Import reaction data from ORD**
CLI: `ord_convert`.
Docs: `user_guide/cli_interface`

## Training and benchmarking

Training is the one area where a GPU is worth setting up.

**Train a ranking or filtering policy**
`PolicyNetworkConfig` → `create_policy_dataset` → `run_policy_training`
(`synplan.ml.training.supervised`).
CLI: `ranking_policy_training`, `filtering_policy_training`.
Tutorial: `04_Policy_Training`
Docs: `methods/policy`, `configuration/policy`

**Train or tune the MHN ranking policy**
`MHNRankingPolicyNetworkConfig` → `create_policy_dataset` →
`run_policy_training` → `run_mhn_network_tuning`.
CLI: `mhn_network_tuning`.
Tutorial: `14_MHN_Ranking_Training`
Docs: `configuration/policy` — "MHN ranking policy"

**Train a value network**
CLI: `value_network_tuning`.
Docs: `methods/value`, `configuration/value`

**Benchmark planning performance**
Docs: `configuration/policy` — "Benchmark recipe". Colab:
`colab/planning_benchmarking.ipynb`.

**Log training runs**
Optional extras `wandb` / `mlflow` / `loggers`.
Docs: `configuration/policy` — "Training logger"

## When old code stops working

**Code or SMARTS from an older SynPlanner behaves differently**
Read `user_guide/migration` before debugging. It covers: chython 1.100 changing
what SMARTS mean, route post-processing moving to
`synplan.chem.reaction.routes`, reaction rules moving to
`synplan.chem.reaction.rules`, tree persistence and route exports, per-node state
moving off `Tree`, and `Tree.stats` becoming a typed dataclass.
Docs: `user_guide/migration`
