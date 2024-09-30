.. _download:

================================
Data download
================================

All SynPlanner data can be downloaded with corresponding  CLI command from HugginFace repository. Here the description of data acoomopnaying SynPlanner.

| 📁 **uspto** - reaction data source
| ``uspto/uspto_standardized.smi`` - the USPTO dataset from the study of Lin et al.
| ``uspto_reaction_rules.pickle`` - the reaction rules were extracted with SynPlanner from the standardized USPTO dataset.
| ``weights/ranking_policy_network.ckpt`` - needed for the prediction of applicable reaction rules in node expansion step in MCTS retrosynthetic planning. Ranking policy network trained on the standardized and filtered USPTO dataset and corresponding extracted reaction rules.
| ``weights/filtering_policy_network.ckpt`` - needed for the prediction of applicable reaction rules in node expansion step in MCTS retrosynthetic planning. Filtering policy network trained on the data extracted from the ChEMBL database and corresponding extracted reaction rules.
| ``weights/value_network.ckpt`` - needed for the prediction of synthesizability of precursors in node evaluation step in MCTS retrosynthetic planning. Value network trained on the planning simulations results for molecules extracted from the ChEMBL database and corresponding trained ranking policy network with building blocks.

| 📁 **chembl** - molecule data source
| ``molecules_for_filtering_policy_training.smi`` - the dataset of molecules extracted from the ChEMBL database for filtering policy network training with extracted reaction rules. Each reaction rule is applied to each molecule, and successfully applied reaction rules are labeled for network training
| ``targets_for_value_network_training.smi`` - the dataset of molecules extracted from the ChEMBL database is used as targets for planning simulations in value network tuning.

| 📁 **building_blocks** - building blocks
| ``building_blocks_em_sa_ln.smi`` - the collection of building blocks for retrosynthetic planning. If during the planning the generated precursors are found in building blocks, the planning is terminated successfully. The default collection is the eMolecules and Sigma Aldrich building block datasets from the ASKCOS tool. The building blocks were standardized by SynPlanner.

| 📁 **benchmarks** - SynPlanner original benchmarks
| ``sascore`` - the Synthetic Accessibility Score benchmark dataset (SAScore benchmark) addresses the retrosynthetic planning for molecules of different synthetic complexities. The dataset consists of 700 target molecules split into seven equal-size subsets (100 molecules each) corresponding to synthetic accessibility score (SAScore, calculated with the RDKit package) 1-2, 2-3, 3-4, etc., up to the maximal value of 9 achieved in the dataset.

| 📁 **tutorial** - the data for SynPlanner tutorials.
| Tutorial data can be used as input data for any SynPlanner tutorial so that the separate SynPlanner pipeline steps are accessible separately (it is not needed to reproduce the pipeline from scratch).
| ``data_curation/uspto_standardized.smi`` - the USPTO dataset is used as input data for the data curation tutorial in SynPlanner.
| ``data_curation/uspto_filtered.smi`` - the standardized and filtered USPTO dataset is ready for reaction rules extraction by the corresponding tutorial.
| ``rules extraction/uspto_reaction_rules.pickle`` - the reaction rules were extracted from the USPTO dataset. Can be used for ranking and filtering policy training in policy training tutorials.
| ``ranking_policy_training/ranking_policy_dataset.pt`` - the training set was created from filtered USPTO and extracted reaction rules for ranking policy training.
| ``ranking_policy_training/ranking_policy_network.ckpt`` - the trained ranking policy network, that can be used in retrosynthetic planning
| ``uspto_tutorial.smi`` - the reduced version of the USPTO dataset was created for demonstrative and educational purposes. The reproduction of the SynPlanner pipeline (starting from data curation) with the reduced version should take around 1 hour and should be feasible on regular machines with limited computation power.
