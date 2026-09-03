# SynPlanner tutorials

Every tutorial here runs in Google Colab — click a badge and it installs
SynPlanner and downloads the public data it needs. Tutorial 20 additionally
requires the prepared vendor-aware InChIKey JSON catalogue to be uploaded or
mounted.

Locally they run the same way once SynPlanner is installed; the setup cell at the
top of each notebook does nothing outside Colab.

## Getting started

| Tutorial | What it covers | |
| --- | --- | --- |
| [Welcome to chython](00_Welcome_to_Chython.ipynb) | the molecule and reaction objects everything else is built on | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/00_Welcome_to_Chython.ipynb) |
| [Coming from RDKit](01_Coming_from_RDKit.ipynb) | the same operations side by side, and where the two libraries disagree | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/01_Coming_from_RDKit.ipynb) |

## The pipeline

| Tutorial | What it covers | |
| --- | --- | --- |
| [Data curation](02_Data_Curation.ipynb) | standardise and filter a reaction dataset before extracting rules from it | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/02_Data_Curation.ipynb) |
| [Rules extraction](03_Rules_Extraction.ipynb) | turn curated reactions into the retrosynthetic rules the search applies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/03_Rules_Extraction.ipynb) |
| [Policy training](04_Policy_Training.ipynb) | train the ranking and filtering networks that choose which rule to try | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/04_Policy_Training.ipynb) |
| [Retrosynthetic planning](05_Retrosynthetic_Planning.ipynb) | run a search, read the routes it found, rank them, save the search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/05_Retrosynthetic_Planning.ipynb) |
| [Tree analysis](06_Tree_Analysis.ipynb) | read a saved search back: policy performance, branching, where routes appeared | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/06_Tree_Analysis.ipynb) |
| [Clustering](07_Clustering.ipynb) | group routes by strategy using RouteCGRs and strategic-bond CGRs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/07_Clustering.ipynb) |
| [Protection scoring](08_Protection_Scoring.ipynb) | find competing functional groups and score routes for selectivity trouble | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/08_Protection_Scoring.ipynb) |

## Going further

| Tutorial | What it covers | |
| --- | --- | --- |
| [Combined ranking and filtering policy](09_Combined_Ranking_Filtering_Policy.ipynb) | use both networks together | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/09_Combined_Ranking_Filtering_Policy.ipynb) |
| [NMCS algorithms](10_NMCS_Algorithms.ipynb) | nested Monte Carlo search, and whether it finds different routes than UCT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/10_NMCS_Algorithms.ipynb) |
| [Planning with RDKit](11_Planning_with_RDKit.ipynb) | hand RDKit molecules in and get RDKit molecules back | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/11_Planning_with_RDKit.ipynb) |
| [Rule analysis](12_Rule_Analysis.ipynb) | inspect and draw the extracted reaction rules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/12_Rule_Analysis.ipynb) |
| [Priority rules](13_Priority_Rules.ipynb) | steer a search with retrosynthetic SMARTS you wrote yourself | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/13_Priority_Rules.ipynb) |
| [MHN ranking training](14_MHN_Ranking_Training.ipynb) | train and fine-tune the modern Hopfield ranking policy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/14_MHN_Ranking_Training.ipynb) |
| [Compare route sets](15_Routes_compare.ipynb) | overlap between two searches, and against another program's routes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/15_Routes_compare.ipynb) |
| [Building block search](16_Building_block_search.ipynb) | which catalogue molecules a route set actually reaches for | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/16_Building_block_search.ipynb) |
| [Synthon-based library design](17_Synthon_Based_Design.ipynb) | fragment molecules into synthons, enumerate libraries and analogues | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/17_Synthon_Based_Design.ipynb) |
| [Retrosynthesis with synthon priority rules](18_Retrosynthesis_With_Synthon_Priority_Rules.ipynb) | the shipped synthon disconnections as a curated rule set | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/18_Retrosynthesis_With_Synthon_Priority_Rules.ipynb) |
| [InChIKey building-block catalogue](20_InChIKey_Building_Block_Catalogue.ipynb) | run connectivity-only MCTS for Boceprevir and calculate vendor costs from a prepared JSON catalogue | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/20_InChIKey_Building_Block_Catalogue.ipynb) |

## Data used by the tutorials

Most tutorials call `download_preset("synplanner-gps")`, which fetches models,
rules and a building-block catalogue from HuggingFace into `synplan_data/`. Files
that are small enough to keep in the repository live in `data/`, and a notebook
downloads them when it is running somewhere the repository is not. Tutorial 20
uses the preset models and rules but requires a separately prepared InChIKey JSON
catalogue.
