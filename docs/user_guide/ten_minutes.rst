10 minutes to SynPlanner
========================

``SynPlanner`` is an end-to-end tool for retrosynthetic planning.
It combines reaction rule extraction, neural network training, and Monte Carlo Tree Search (MCTS)
to find synthetic routes for target molecules.

.. note::
    **Coming from RDKit?** SynPlanner uses `chython <https://github.com/chython>`_ for chemistry
    instead of RDKit. If you are used to RDKit's API, see the
    :doc:`01_Coming_from_RDKit` tutorial for a side-by-side comparison before continuing here.

Step 1: Installation
--------------------

.. code-block:: bash

    pip install SynPlanner

See :doc:`../get_started/installation` for GPU support and conda-based setup.

Step 2: Download data
---------------------

Use the preset downloader to fetch pre-trained models, reaction rules, and building blocks
from HuggingFace.

.. code-block:: bash

    synplan download_preset --preset synplanner-gps --save_to synplan_data

Or from Python:

.. code-block:: python

    from synplan.utils.loading import download_preset

    paths = download_preset("synplanner-gps", save_to="synplan_data")

After downloading, you will have:

.. code-block:: text

    synplan_data/
    ├── policy/supervised_gps/v1/
    │   ├── reaction_rules.tsv          # reaction rules
    │   └── v1/
    │       └── ranking_policy.ckpt     # ranking policy network weights
    ├── policy/supervised_gcn/v1/v1/
    │   └── filtering_policy.ckpt       # filtering policy network weights
    ├── value/supervised_gcn/v1/
    │   └── value_network.ckpt          # value network weights (advanced)
    └── building_blocks/emolecules-salt-ln/
        └── building_blocks.tsv         # purchasable building blocks

The rules and ranking head of ``synplanner-gps`` live under ``supervised_gps``;
only its filtering head comes from ``supervised_gcn``. Use
``--preset synplanner-article`` if you need a filtering and a ranking head
trained on the same rule set (see :doc:`09_Combined_Ranking_Filtering_Policy`).

The CLI prints the local path of every file it downloaded — prefer copying those
paths over retyping them.

Step 3: Key concepts
--------------------

Before running planning, it helps to understand what each component does:

**Reaction rules**
    Extracted reaction templates (SMARTS patterns) encoding known chemical transformations.
    The policy network ranks them for each molecule during the search.

**Building blocks**
    Purchasable molecules that terminate a retrosynthetic route.
    A route is considered solved when all precursors are found in this set.

**Policy network**
    A graph neural network that predicts which reaction rules are applicable to a given molecule,
    ranked by predicted probability. Two types are available:

    - *Ranking policy*: ranks all applicable rules.
    - *Filtering policy*: filters out unlikely rules before ranking (faster for large rule sets).

**Evaluation function**
    Estimates the retrosynthetic feasibility of a newly created precursor node.
    Two main types:

    - *Rollout* (default): performs a short forward simulation to estimate feasibility.
      No additional training required; works with any building block set.
    - *Value network* (advanced): a trained GCN that instantly predicts feasibility.
      Faster per evaluation but requires a separate training step.

**Search tree (MCTS)**
    The tree explores retrosynthetic pathways by iteratively selecting, expanding,
    evaluating, and backpropagating nodes. Planning stops when a route to building blocks
    is found, or when the iteration/time limit is reached.

Step 4: Plan via Python API
---------------------------

The example below uses the ranking policy network and rollout evaluation. This is the default configuration, requiring no value network.

.. code-block:: python

    from synplan.utils.loading import (
        load_building_blocks, load_reaction_rules,
        load_policy_function, load_evaluation_function,
        download_preset,
    )
    from synplan.mcts.config import TreeConfig, RolloutEvaluationConfig
    from synplan.chem.utils import mol_from_smiles
    from synplan.mcts.tree import Tree

    # Download preset data (skip if already downloaded)
    paths = download_preset("synplanner-gps", save_to="synplan_data")

    # Load components (preset building blocks are already standardized)
    building_blocks = load_building_blocks(paths["building_blocks"], standardize=False)
    reaction_rules  = load_reaction_rules(paths["reaction_rules"])
    policy_network  = load_policy_function(weights_path=paths["ranking_policy"])

    # Configure the search tree
    tree_config = TreeConfig(
        search_strategy="expansion_first",  # recommended with rollout evaluation
        max_iterations=300,                 # increase for harder targets
        max_time=120,                       # seconds; increase for harder targets
        max_depth=9,                        # maximum retrosynthetic depth
        min_mol_size=6,                     # molecules smaller than this are treated as building blocks
        init_node_value=0.5,                # initial node value for expansion_first strategy
        ucb_type="uct",
        c_ucb=0.1,
    )

    # Set up rollout evaluation
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_network,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
    )
    evaluation_function = load_evaluation_function(eval_config)

    # Define the target molecule (capivasertib, an anti-cancer drug)
    target = mol_from_smiles(
        "NC1(C(=O)N[C@@H](CCO)c2ccc(Cl)cc2)CCN(c2nc[nH]c3nccc2-3)CC1",
        standardize=True,
    )

    # Run the search
    tree = Tree(
        target=target,
        config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_network,
        evaluation_function=evaluation_function,
    )

    found_routes = []
    for solved, node_id in tree:
        if solved:
            found_routes.append(node_id)
            if len(found_routes) >= 5:
                break

    print(tree)  # summary: nodes explored, routes found, time elapsed

.. note::
    Pass ``standardize=True`` only for your own, un-standardized building-block
    file. It runs a process pool, so on macOS and Windows the calling code must
    sit under ``if __name__ == "__main__":`` — otherwise Python's spawn start
    method re-imports the script and raises
    ``RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase``.

Inspect and visualise routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the search, extract routes and generate an HTML report:

.. code-block:: python

    from synplan.utils.visualisation import routes_report_html

    routes = tree.routes()          # list of Route objects, best score first
    print(f"Found {len(routes)} route(s)")

    # Save a self-contained HTML report of exactly those routes
    routes_report_html(routes, html_path="routes.html")

Score routes for protection group issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The protection scoring module flags competing functional groups in a route: steps where
a reagent may react with an unintended site, indicating a potential need for protecting groups.
This follows the methodology of `Westerlund et al. (2026) <https://doi.org/10.1021/acs.jcim.6c01147>`_.

Ranking is a second step, after the search:

.. code-block:: python

    from synplan.chem.reaction.routes.quality.scorer import ProtectionRouteScorer

    # Build scorer with default configuration (bundled data). Keep it: it holds the
    # cache for 102 SMARTS patterns, which a scorer built per call would throw away.
    route_scorer = ProtectionRouteScorer.from_config()

    best = route_scorer.rank(tree.routes())     # best first
    routes_report_html(best[:5], html_path="best.html")

The score S(T) is in [0, 1]: 1.0 means no competing interactions detected,
lower values indicate steps that may require protecting group strategies.
``CompetingSitesRouteScorer.from_config().score(route)`` returns it for one route. The scan is expensive —
about 64 ms per molecule the cache has not seen — which is why the search does not
do it, ``tree.routes()`` does not either, and ``rank`` costs tens of seconds on a
few hundred routes rather than the instant its ``sorted``-like name suggests.

``rank`` orders by whatever the scorer's ``score`` says, and for this scorer that
is ``search_score * S(T)``, the re-ranking of the paper. It is therefore defined
only for routes a search produced: a route read back out of a file carries no
search score, and ``score`` says so rather than inventing one. Rank those with
``CompetingSitesRouteScorer``, which judges a route on its own terms::

    from synplan.chem.reaction.routes.quality.scorer import CompetingSitesRouteScorer

    from_file = CompetingSitesRouteScorer.from_config().rank(routes)

See :doc:`08_Protection_Scoring` for a detailed walkthrough.

Batch planning (many targets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For running many targets and saving results to disk, use the higher-level ``run_search``
function, which mirrors the CLI behaviour:

.. code-block:: python

    from synplan.mcts.search import run_search
    from synplan.ml.config import PolicyNetworkConfig
    from synplan.mcts.config import RolloutEvaluationConfig

    run_search(
        targets_path="targets.smi",
        search_config=tree_config.to_dict(),
        policy_config=PolicyNetworkConfig(weights_path=paths["ranking_policy"]),
        evaluation_config=RolloutEvaluationConfig(
            policy_network=policy_network,
            reaction_rules=reaction_rules,
            building_blocks=building_blocks,
            min_mol_size=tree_config.min_mol_size,
            max_depth=tree_config.max_depth,
        ),
        reaction_rules_path=paths["reaction_rules"],
        building_blocks_path=paths["building_blocks"],
        results_root="planning_results",
        route_scorer=route_scorer,  # optional
    )

Results are written to ``planning_results/``: ``tree_search_stats.csv``, ``extracted_routes.json``,
and per-target HTML files in ``extracted_routes_html/``.

Key parameters to tune
~~~~~~~~~~~~~~~~~~~~~~~

.. table::
    :widths: 30 70

    ========================= ===============================================================
    Parameter                 When to change
    ========================= ===============================================================
    ``max_iterations``        Increase (e.g. 500–1000) for complex targets with many
                              possible disconnections.
    ``max_time``              Hard wall-clock limit in seconds. Useful for batch jobs.
    ``max_depth``             Increase beyond 9 for targets requiring many synthetic steps.
    ``min_mol_size``          Lower to 1 to allow very small fragments as building blocks;
                              raise to filter out trivial disconnections.
    ``search_strategy``       Use ``expansion_first`` with rollout (default).
                              Use ``evaluation_first`` only with a value network.
    ``evaluation_type``       ``rollout`` (default, no extra training) or
                              ``gcn`` (value network, faster but needs training).
    ``top_rules``             Reduce (e.g. 20) for speed; increase for broader search.
    ``rule_prob_threshold``   Raise (e.g. 0.1) to skip low-probability rules and speed
                              up the search.
    ========================= ===============================================================

Step 5: Plan via CLI
--------------------

Batch-plan a list of targets from the command line:

.. code-block:: bash

    synplan planning \
      --config configs/planning_standard.yaml \
      --targets targets.smi \
      --reaction_rules synplan_data/policy/supervised_gps/v1/reaction_rules.tsv \
      --building_blocks synplan_data/building_blocks/emolecules-salt-ln/building_blocks.tsv \
      --policy_network synplan_data/policy/supervised_gps/v1/v1/ranking_policy.ckpt \
      --results_dir planning_results

``targets.smi`` is a plain text file with one SMILES per line — you write it
yourself, nothing ships one. ``configs/planning_standard.yaml`` comes from the
git repository; ``pip install SynPlanner`` does not install the ``configs/``
directory (see :doc:`../get_started/installation`).
Results are written to ``planning_results/``: a CSV with per-target statistics,
JSON routes, and HTML visualisations.

To use the value network for evaluation (advanced), switch the config as well:

.. code-block:: bash

    synplan planning \
      --config configs/planning_value.yaml \
      --value_network synplan_data/value/supervised_gcn/v1/value_network.ckpt \
      ...   # same --targets/--reaction_rules/--building_blocks/--policy_network

.. warning::
    ``--value_network`` alone does nothing. The evaluator is chosen by
    ``node_evaluation: evaluation_type:`` in the config file, and
    ``planning_standard.yaml`` omits that section, so it defaults to
    ``rollout``. Adding ``--value_network`` to the standard command runs to
    completion, prints the same "solved" summary, and never loads the value
    network. ``configs/planning_value.yaml`` sets ``evaluation_type: gcn``
    (and ``search_strategy: evaluation_first``, which is the strategy the value
    network is meant for).

See :doc:`cli_interface` for a full list of all CLI commands and options.

Step 6: Visualise results
-------------------------

After planning, open the HTML report:

.. code-block:: bash

    open planning_results/extracted_routes_html/retroroutes_target_0.html

Or generate visualisations from Python:

.. code-block:: python

    from synplan.utils.visualisation import routes_report_html

    routes_report_html(tree.routes(), html_path="routes.html")

Next steps
----------

**Go deeper into planning:**

- :doc:`05_Retrosynthetic_Planning`: detailed planning tutorial with route inspection
- :doc:`../configuration/planning`: all planning configuration parameters
- :doc:`../methods/mcts`: how MCTS and search strategies work

**Train your own models on custom data:**

- :doc:`02_Data_Curation`: prepare reaction data
- :doc:`03_Rules_Extraction`: extract reaction rules
- :doc:`04_Policy_Training`: train ranking and filtering policy networks
- :doc:`../configuration/policy`: policy network configuration

**Advanced search:**

- :doc:`09_Combined_Ranking_Filtering_Policy`: combine ranking and filtering policies
- :doc:`10_NMCS_Algorithms`: Nested Monte Carlo Search
- :doc:`../methods/value`: value network concepts

**Route analysis:**

- :doc:`06_Tree_Analysis`: diagnose policy and search behaviour
- :doc:`07_Clustering`: cluster routes by strategic bonds
- :doc:`08_Protection_Scoring`: score routes for selectivity issues
