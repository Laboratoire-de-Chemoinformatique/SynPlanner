Data
====

This section summarizes the datasets used by SynPlanner and how to obtain them.

Overview of datasets
--------------------

``SynPlanner`` operates reaction and molecule data stored in different formats.

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Data type                          Description
    ================================== =================================================================================
    Reactions                          Reactions can be loaded and stored as the list of reaction smiles in the file (.smi) or RDF File (.rdf)
    Molecules                          Molecules can be loaded and stored as the list of molecule smiles in the file (.smi) or SDF File (.sdf)
    Reaction rules                     Reaction rules can be loaded and stored as the pickled list of CGRtools ReactionContainer objects (.pickle)
    Retrosynthetic models              Retrosynthetic models (neural networks) can be loaded and stored as serialized PyTorch models (.ckpt)
    Retrosynthetic routes              Retrosynthetic routes can be visualized and stored as HTML files (.html) and can be stored as JSON files (.json)
    ================================== =================================================================================

.. note::
    Reaction and molecule file formats are parsed and recognized automatically by ``SynPlanner`` from file extensions.
    Be sure to store the data with the correct extension.

Data sources and bundles
------------------------

| 📁 **uspto** – reaction data source
| ``uspto/uspto_standardized.smi`` – the USPTO dataset from the study of Lin et al.
| ``uspto_reaction_rules.pickle`` – reaction rules extracted with SynPlanner from the standardized USPTO dataset.
| ``weights/ranking_policy_network.ckpt`` – trained on standardized and filtered USPTO and the corresponding rules.
| ``weights/filtering_policy_network.ckpt`` – trained on ChEMBL-derived data and corresponding rules.
| ``weights/value_network.ckpt`` – trained from planning simulations on ChEMBL and a trained policy network.

| 📁 **chembl** – molecule data source
| ``molecules_for_filtering_policy_training.smi`` – ChEMBL molecules for filtering policy training (rule applications labeled).
| ``targets_for_value_network_training.smi`` – ChEMBL targets used for value network tuning simulations.

| 📁 **building_blocks** – building blocks
| ``building_blocks_em_sa_ln.smi`` – standardized building blocks (eMolecules + Sigma Aldrich from ASKCOS).

| 📁 **benchmarks** – SynPlanner original benchmarks
| ``sascore`` – SAScore benchmark (700 targets split into 7 subsets by SAScore).

| 📁 **tutorial** – tutorial input data
| ``data_curation/uspto_standardized.smi`` – input for the data curation tutorial.
| ``data_curation/uspto_filtered.smi`` – standardized and filtered USPTO for rule extraction.
| ``rules extraction/uspto_reaction_rules.pickle`` – extracted rules for policy training tutorials.
| ``ranking_policy_training/ranking_policy_dataset.pt`` – dataset for ranking policy training.
| ``ranking_policy_training/ranking_policy_network.ckpt`` – trained ranking policy network.
| ``uspto_tutorial.smi`` – reduced USPTO subset for quick pipeline reproduction.

Download data
-------------

.. include:: ../get_started/data_download.rst

Download from Hugging Face (browse)
-----------------------------------

- Repository root: `Hugging Face – SynPlanner <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main>`_
- Subfolders:
  - `building_blocks/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main/building_blocks>`_
  - `uspto/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main/uspto>`_
  - `weights/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main/weights>`_
  - `benchmarks/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main/benchmarks>`_
  - `tutorial/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main/tutorial>`_


