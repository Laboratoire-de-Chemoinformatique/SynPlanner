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
`MappingConfig` + `map_reactions_from_file` (`synplan.chem.reaction.curation.mapping`), then
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
(`...rules.representation`). `RuleSet.from_tsv` keeps popularity;
`load_reaction_rules` reads the same file and drops it, so use the former when you
want counts. `ruleset[:10]` renders a fixed three-column view; `to_dataframe()`
returns a `ChemFrame` you can filter and sort — slice the RuleSet before drawing,
laying out all 11k rules is slow and renders enormous.
Tutorial: `12_Rule_Analysis`
Docs: `methods/extraction`

**Read the shipped building-block TSV**
`SMILES` plus `LN_ppg`, `SA_ppg`, `EM_ppg`, which are supplier price columns and are
documented nowhere in the codebase. Every one of the 186,868 rows carries a nonzero
figure in at least one of them, so treating the whole file as orderable is safe. The
synthon commands read the file unmodified — they want a headered TSV with a `SMILES`
column and ignore the rest.

**Use my own building blocks**
CLI: `building_blocks_standardizing`. Preset building blocks are already
standardized — pass `standardize=False` to `load_building_blocks`.
Docs: `user_guide/cli_interface`, `user_guide/data`

**Show rules, synthons or any chython objects as a table**
`rules_frame` and `synthons_frame` (`synplan.chem.synthon.frames`),
`tree_stats_frame` (`synplan.utils.frames`). They return a `ChemFrame`: a pandas
frame that draws any column holding something with a `depict()` method. Building a
DataFrame of depictions by hand is the common reinvention here — do not.
`ChemFrame(rows, depict_columns=["reaction"])` takes anything: molecules,
reactions, CGRs, SMARTS queries and synthons all draw with no extra code.
`.df` gives plain pandas with the objects themselves still in the cells, and is
what `.groupby` and `.str` need; a mask or `.head()` returns a ChemFrame and keeps
drawing. The view stops at `max_display_rows` (20) because a drawn row costs
roughly 7 kB of SVG.
Three traps: `rules_frame()` gives all 154 shipped records, `rules_frame(rules)`
restricts and reorders to the loaded ones; `kind` collapses macro over ring, so
every ring-forming rule is `kind != "acyclic"`, not `== "ring"`; and a ring rule's
`rule` and `smarts` show its hand-authored reagent form, not the raw two-cut SMARTS.
Module: `synplan.utils.frames`, `synplan.chem.synthon.frames`
Tutorial: `17_Synthon_Based_Design`, `18_Retrosynthesis_With_Synthon_Priority_Rules`

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

## Synthons (the Synt-On port)

**Turn a building-block catalogue into a synthon stock**
`BBClassifier` assigns one or more of 147 ordered classes, then `BBSynthoniser`
runs that class's rule program. CLI: `bb_classifying` → `bb_synthonizing`.
Config: `synthonisation.yaml` → `SynthonConfig`.
Module: `synplan.chem.synthon`
Tutorial: `17_Synthon_Based_Design`. Docs: `configuration/synthonisation`

**Cut a target into purchasable synthons**
`Fragmenter.fragment` builds a disconnection DAG. The default `rule_mode` of
use_all cuts with 115 rules — 39 acyclic disconnections and 76 ring-forming
ones — plus their macrocyclic twins when the target has a ring larger than 11
atoms. Emptying `ring_closure_sizes` drops the ring rules and restores
acyclic-only behaviour. `rules_selection` is read only when `rule_mode` is
something else, and the shipped R1-R13 names no ring rule.
CLI: `synthon_fragment --stock`.
Module: `synplan.chem.synthon.fragment`
Tutorial: `17_Synthon_Based_Design`

**Recombine stocked synthons into new molecules**
`Enumerator.enumerate_library` grows products from the whole stock with no target;
`Enumerator.enumerate_analogues` fills the slots of one target's fragmentation
pathway. Only the second is wired to the CLI — `synthon_enumerate` calls
`enumerate_file`, which never reaches `enumerate_library`, so a target-free library
is Python only. Do not fake a pathways TSV to force it through the CLI.
Three traps. `max_products` truncates a depth-first walk rather than sampling, so a
capped run returns elaborations of the first seed — measured 300 of 300 products
sharing one seed and one partner — bound the synthon pool instead and let it finish.
`max_reacted_synthons` is what decides whether the run returns at all: 2 exhausts a
few-hundred-synthon pool in a minute, 3 costs roughly thirty times that.
`ro2_filtration` is applied by `load_synthon_stock`, not by the enumerator, so it
does nothing unless the config reaches the loader.
Library mode yields bare molecules with no record of which blocks went in;
`enumerate_analogues` writes the source synthons per row and library mode has no
equivalent, so a shopping list needs building yourself. Re-fragmenting a product to
recover it does not work — the enumerator and the fragmenter are not inverses.
CLI: `synthon_enumerate` (analogues only).
Module: `synplan.chem.synthon.enumerate`
Tutorial: `17_Synthon_Based_Design`

**Make analogues of a hit that are actually purchasable**
Fragment the hit, then `SynthonStock.slots()` for candidates per slot, then
`Enumerator.enumerate_analogues`. Four things decide whether this works at all.
`find_analogues` defaults to off and it IS the feature — off, every slot offers
only the hit's own synthon and you rebuild the hit. `strict_availability` is off
by default too, and then an unfillable slot silently falls back to the hit's own synthon,
which need not be purchasable, so the shelf guarantee quietly breaks.
`mw_lower`/`mw_upper` bound `enumerate_library` ONLY — `enumerate_analogues` ignores
them, so they will not keep an analogue library drug-sized.
And `best_available()[0]` is not guaranteed fillable; walk down the list until one
yields products, because zero-with-full-slots and zero-with-empty-slots look the same.
Two semantics to state to anyone reading the output. `is_analogue` matches on ring
count, heavy-atom count and element census, and two of its four branches never test
substructure at all — over a real catalogue it delivers scaffold HOPS, not periphery
decoration, and `similarity_threshold` cannot tighten it because the Tanimoto branch
is unioned with this one. Reassembly joins by label compatibility alone and does not
remember which label was bonded to which, so a pharmacophore can be scrambled away —
measured, 1440 sorafenib products none of which kept the urea. Pin the slot carrying
the pharmacophore (`slots[core] = [core]`, only if that synthon is stocked as itself)
and filter the products on a core SMARTS.
Do not confuse `find_analogues` the config flag with `find_analogues()` the function —
the function returns SYNTHONS, not molecules, and is exported at package top level.
Module: `synplan.chem.synthon.analogues`, `synplan.chem.synthon.stock`
Tutorial: `17_Synthon_Based_Design`

**Use the synthon disconnections during planning**
`synthon_priority_rules()` returns them as `run_search(priority_rules=...)` input
under the source name `"synthon"`; set `use_priority=True` in the search config
or they are ignored. The children are ordinary molecules against the ordinary
building-block stock — there is no synthon stock in the tree. A ring rule loads
only when it ships a hand-authored `retro_smarts` naming its real reagents, since
capping cannot spell a leaving group for a two-bond cut; 69 of the 76 do, so the
default set is 39 acyclic plus 69 ring rules.
Module: `synplan.chem.reaction.rules.synthon`
Tutorial: `18_Retrosynthesis_With_Synthon_Priority_Rules`

**Drop reactions the synthon rules already cover from a training corpus**
`classify_coverage` says whether a mapped reaction builds a bond one of the 39
acyclic disconnections breaks, with the reactant-side leaving groups checked
against the rule's labels. CLI: `synthon_coverage --keep uncovered|covered`.
37.9% of a 100k USPTO sample is covered.
Module: `synplan.chem.synthon.coverage`

**Catalogue analysis**
`scaffold_smiles` and `murcko_scaffold` take a Bemis-Murcko scaffold after removing
ring-containing protecting groups — do not substitute plain RDKit Murcko, which keeps
the Cbz and drops an amide carbonyl; `ro2_pass` and `ro2_filter` apply the rule of two.
`scaffolds_file` runs a whole catalogue but is single-threaded, unlike its neighbours.
CLI: `bb_scaffolds` (a command, not a Python symbol).
Module: `synplan.chem.scaffolds`, `synplan.chem.synthon.stock`,
`synplan.interfaces.synthon_commands`

**Judge a catalogue before buying it**
`classify_file` then `synthonise_file` (`synplan.interfaces.synthon_commands`) with
`write_audit_files: true`, then read `summary.json`: the share of rows carrying no
reactive class is the number a purchase decision turns on. Run it in audit mode or the
figures lie — outside audit mode `classify_file` silently drops unclassified rows and
you cannot tell those from parse failures. `load_synthon_stock` returns
`synthon -> {blocks}`, so reactive redundancy falls out of it; exact-SMILES dedupe
measures nothing. Note `classify_file` returns an int and `synthonise_file` a pair.
CLI: `bb_classifying` → `bb_synthonizing`.
Module: `synplan.chem.synthon.stock`

**Keep an auditable record of any synthon CLI run**
All five audited synthon commands accept `synthonisation.yaml`
(`synthon_coverage` takes the config but writes no sidecars). Set
`write_audit_files: true` to create `fallback.smi`, `fallback.tsv`,
`errors.tsv`, `summary.json`, and `run.log` beside the requested output. Use a
dedicated directory per command because those sidecar names are fixed.
`fallback.smi` contains only valid retryable inputs; for `synthon_enumerate` it
preserves the complete fragmentation TSV row. `fallback.tsv` also records
processing errors. Metadata on SMI/CXSMILES records must be separated by TAB.

**Regenerate the shipped synthon data**
`python -m synplan.chem.synthon.rules._convert <Synt-On/config> --out
synplan/chem/synthon/rules --check`. The JSON is committed; nothing translates at
import time.

**Say what the disconnections do not cover**
Every rule records its provenance: 78 converted from the reference, 76 ring rules
authored in-repo and not yet signed off by a chemist. Ten more are held out of
`rules.json` entirely — `chemist_review` in the development docs is the queue,
and the port has no rule for a plain 2,5-dialkylfuran because two of them are
held. The pipeline is also stereo-blind: `safe_canonicalization` discards
stereocentres, so any route through a ring rule is racemic. Promote
`StereoDiscardedWarning` to an error to refuse stereo-bearing input rather than
racemise it silently.
Module: `synplan.chem.utils`

## When old code stops working

**Code or SMARTS from an older SynPlanner behaves differently**
Read `user_guide/migration` before debugging. It covers: chython 1.100 changing
what SMARTS mean, route post-processing moving to
`synplan.chem.reaction.routes`, reaction rules moving to
`synplan.chem.reaction.rules`, tree persistence and route exports, per-node state
moving off `Tree`, and `Tree.stats` becoming a typed dataclass.
Docs: `user_guide/migration`
