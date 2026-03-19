<p align="center">
  <img src="https://raw.githubusercontent.com/Laboratoire-de-Chemoinformatique/SynPlanner/main/docs/images/banner.png" alt="SynPlanner banner" width="700">
</p>

<h3 align="center">End-to-end retrosynthetic planning from raw reaction data</h3>

<p align="center">
  <a href="https://synplanner.readthedocs.io/"><img src="https://img.shields.io/badge/docs-readthedocs-blue" alt="Docs"></a>
  <a href="https://synplanner.readthedocs.io/en/latest/user_guide/index.html"><img src="https://img.shields.io/badge/tutorials-12_notebooks-orange" alt="Tutorials"></a>
  <a href="https://doi.org/10.1021/acs.jcim.4c02004"><img src="https://img.shields.io/badge/paper-JCIM_2025-green" alt="Paper"></a>
  <a href="https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner"><img src="https://img.shields.io/badge/demo-Hugging_Face-yellow" alt="GUI demo"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/SynPlanner/"><img src="https://img.shields.io/pypi/v/SynPlanner.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/SynPlanner/"><img src="https://img.shields.io/badge/python-3.10--3.14-blue" alt="Python"></a>
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner" alt="License"></a>
  <a href="https://pepy.tech/projects/synplanner"><img src="https://img.shields.io/pepy/dt/SynPlanner" alt="Downloads"></a>
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/commits"><img src="https://img.shields.io/github/last-commit/Laboratoire-de-Chemoinformatique/SynPlanner" alt="Last commit"></a>
</p>

---

`SynPlanner` is an open-source tool for retrosynthetic planning.
It integrates Monte Carlo Tree Search (MCTS) with graph neural networks
to evaluate applicable reaction rules (policy network) and
the synthesizability of intermediate products (value network).

- **Data curation:** standardize and filter raw chemical reaction data
- **Rule extraction:** extract reaction templates with configurable specificity
- **Model training:** train policy and value networks (supervised + RL)
- **Retrosynthesis:** MCTS-based planning with multiple search strategies
- **Route quality:** competing-sites scoring for functional group selectivity ([Westerlund et al.](https://chemrxiv.org/engage/chemrxiv/article-details/68ff62f6ef936fb4a29b02d8))
- **Route clustering:** group routes by strategic bonds ([Gilmullin et al.](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-lnkz6-vz))
- **Visualization:** HTML route reports and interactive GUI

## Installation

**Requires:** Python 3.10 – 3.14 &middot; Linux x86_64, macOS arm64 &middot; [Docker images](https://synplanner.readthedocs.io/en/latest/get_started/index.html) for other platforms

```bash
pip install SynPlanner
synplan --version
```

## Quick start

**1.** Download pre-trained models, rules, and building blocks:

```bash
synplan download_preset --preset synplanner-article --save_to synplan_data
```

**2.** Run planning on a target molecule:

```bash
synplan planning \
  --config configs/planning_standard.yaml \
  --targets targets.smi \
  --reaction_rules synplan_data/policy/supervised_gcn/v1/reaction_rules.tsv \
  --building_blocks synplan_data/building_blocks/emolecules-salt-ln/building_blocks.tsv \
  --policy_network synplan_data/policy/supervised_gcn/v1/v1/ranking_policy.ckpt \
  --results_dir planning_results
```

> [!TIP]
> Try it in the browser, no installation required:
> [Retrosynthetic planning](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/retrosynthetic_planning.ipynb) &middot;
> [Benchmarking](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/planning_benchmarking.ipynb) &middot;
> [Route clustering](https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/routes_clustering.ipynb)

The full CLI includes commands for every pipeline step: `reaction_mapping`, `reaction_standardizing`, `reaction_filtering`, `rule_extracting`, `ranking_policy_training`, `planning`, `clustering`, and more. Run `synplan --help` for the complete list.

For local tutorials covering all steps from data curation to protection scoring, see the [tutorials/](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials) directory and the [documentation](https://synplanner.readthedocs.io/).

## Team

**Questions & bug reports:** open an [issue](https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/issues) or contact [Tagir Akhmetshin](https://github.com/tagirshin) (lead developer) and [Almaz Gilmullin](https://github.com/Protolaw) (clustering module)

**Contributors:**
[Timur Madzhidov](mailto:tmadzhidov@gmail.com) (initiator),
[Alexandre Varnek](mailto:varnek@unistra.fr) (supervisor),
[Dmitry Zankov](https://github.com/dzankov) (data curation, tutorials, reproducibility),
[Philippe Gantzer](https://github.com/PGantzer) (GUI, writing module),
[Dmitry Babadeev](https://github.com/prog420) (planning, visualization),
[Anna Pinigina](mailto:anna.10081048@gmail.com) (rule extraction),
[Milo Roucairol](https://github.com/RoucairolMilo) (search strategies),
[Mikhail Volkov](https://github.com/mbvolkoff) (testing)

## Citation

If you use `SynPlanner` in your research, please cite:

> Akhmetshin, T.; Zankov, D.; Gantzer, P.; Babadeev, D.; Pinigina, A.; Madzhidov, T.; Varnek, A.
> **SynPlanner: An End-to-End Tool for Synthesis Planning.**
> *J. Chem. Inf. Model.* **2025**, *65* (1), 15–21.
> [doi:10.1021/acs.jcim.4c02004](https://doi.org/10.1021/acs.jcim.4c02004)

If you use route clustering, please also cite:

> Gilmullin, A.; Akhmetshin, T.; Madzhidov, T.; Varnek, A.
> **Route Clustering by Strategic Bonds.**
> *ChemRxiv*, 2025.
> [doi:10.26434/chemrxiv-2025-lnkz6-vz](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-lnkz6-vz)

If you use the protection / route quality scoring, please also cite:

> Westerlund, A. M. et al.
> **Toward Lab-Ready AI Synthesis Plans with Protection Strategies and Route Scoring.**
> *ChemRxiv*, 2025.
> [doi:10.26434/chemrxiv-2025-68ff6](https://chemrxiv.org/engage/chemrxiv/article-details/68ff62f6ef936fb4a29b02d8)
