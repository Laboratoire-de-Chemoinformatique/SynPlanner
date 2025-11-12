10 minutes to SynPlanner
========================

Get Started
-----------

For installation and data download, please see the :doc:`../get_started/index` guide.
After that, you will need:

- Reaction rules (``uspto/uspto_reaction_rules.pickle``)
- Policy network weights (``uspto/weights/ranking_policy_network.ckpt``)
- Building blocks (``building_blocks/building_blocks_em_sa_ln.smi``)

Important recommendations
--------------------------------------------------

- The "evaluation_first" strategy is not recommended with rollout evaluation; use
  ``search_strategy='expansion_first'`` for rollout/random. With a value network (``evaluation_type='gcn'``),
  both strategies work, but we still suggest ``expansion_first`` for speed.
- For complex targets or limited reaction rules/building blocks, increase ``max_iterations`` and/or ``max_time``.
- First building blocks load can be slow if they are standardized from SMILES. The CLI path standardizes on the fly.

Plan via Python API
-------------------

Below is the minimal equivalent of the notebook’s setup. It uses the policy network for expansion
and rollout evaluation (no value network).

.. code-block:: python

   from synplan.utils.loading import load_building_blocks, load_reaction_rules, load_policy_function
   from synplan.utils.config import TreeConfig
   from synplan.chem.utils import mol_from_smiles
   from synplan.mcts.tree import Tree

   # Define paths to data
   reaction_rules_path = "tutorials/synplan_data/uspto/uspto_reaction_rules.pickle"
   building_blocks_path = "tutorials/synplan_data/building_blocks/building_blocks_em_sa_ln.smi"
   policy_weights_path = "tutorials/synplan_data/uspto/weights/ranking_policy_network.ckpt"

   # 1. Load building blocks
   # The first loading can be long, especially from a SMILES file.
   building_blocks = load_building_blocks(building_blocks_path, standardize=True)

   # 2. Load reaction rules
   reaction_rules = load_reaction_rules(reaction_rules_path)

   # 3. Load policy network
   policy_network = load_policy_function(weights_path=policy_weights_path)

   # 4. Configure search
   tree_config = TreeConfig(
       search_strategy="expansion_first",
       evaluation_type="rollout",
       max_iterations=300,
       max_time=120,
       max_depth=9,
       min_mol_size=1,
       init_node_value=0.5,
       ucb_type="uct",
       c_ucb=0.1,
   )

   # Create evaluation configuration
   eval_config = RolloutEvaluationConfig(
       policy_network=policy_network,
       reaction_rules=reaction_rules,
       building_blocks=building_blocks,
       min_mol_size=tree_config.min_mol_size,
       max_depth=tree_config.max_depth,
   )

   # Create evaluator from config
   evaluation_function = load_evaluation_function(eval_config)

   # 5. Load target molecule
   # An example from the tutorial: capivasertib, an anti-cancer medication.
   example_smiles = "NC1(C(=O)N[C@@H](CCO)c2ccc(Cl)cc2)CCN(c2nc[nH]c3nccc2-3)CC1"
   target_molecule = mol_from_smiles(example_smiles, standardize=True)

   # 6. Initialise tree
   tree = Tree(
       target=target_molecule,
       config=tree_config,
       reaction_rules=reaction_rules,
       building_blocks=building_blocks,
       expansion_function=policy_network,
       evaluation_function=evaluation_function,
   )

   # 7. Run search
   found_paths = []
   for solved, node_id in tree:
      if solved:
         print("Solved!")
         found_paths.append(node_id)
         # In a real scenario, you might want to stop after finding a solution
         # or continue searching for more routes.
         if len(found_paths) > 10:
               break
   print(tree)

Plan a few targets (CLI)
------------------------

You can batch-plan targets directly from the terminal using the ``synplan planning`` command.
Parameters mirror those described in the :ref:`cli_interface` reference. In brief:

- ``--config``: planning configuration file.
- ``--targets``: SMILES file with target molecules (one per line).
- ``--reaction_rules``: path to extracted reaction rules.
- ``--building_blocks``: path to standardized building blocks.
- ``--policy_network``: path to a trained ranking or filtering policy network.
- ``--value_network``: optional path to a trained value network.
- ``--results_dir``: directory to write results.

.. code-block:: bash

   synplan planning \
     --config configs/planning.yaml \
     --targets tutorials/synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi \
     --reaction_rules tutorials/synplan_data/uspto/uspto_reaction_rules.pickle \
     --building_blocks tutorials/synplan_data/building_blocks/building_blocks_em_sa_ln.smi \
     --policy_network tutorials/synplan_data/uspto/weights/ranking_policy_network.ckpt \
     --results_dir tutorials/planning_results

Outputs go to ``tutorials/planning_results`` (CSV stats, JSON routes, HTML visualisations).

For a complete overview of commands and parameters, see the :ref:`cli_interface` documentation.
All pipeline steps are available as CLI commands, including data download, building block
standardization, reaction standardization and filtration, reaction rule extraction,
policy network training, value network training, and planning.

Next steps
----------

- Configure planning options: :doc:`../configuration/planning`
- Other steps: :doc:`../user_guide/Step-1_Data_Curation`, :doc:`../user_guide/Step-2_Rules_Extraction`, :doc:`../user_guide/Step-3_Policy_Training`


