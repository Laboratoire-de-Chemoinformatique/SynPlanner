"""Module containing functions for loading reaction rules, building blocks and
retrosynthetic models."""

import contextlib
import functools
import logging
import os
import pickle
import shutil
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Union

import yaml
from chython.files.SDFrw import SDFRead
from chython.reactor.reactor import Reactor
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.utils import (
    AtomMappingCheck,
    reaction_string_mapping_status,
    standardize_sdf_text,
    standardize_smiles_batch,
)
from synplan.ml.networks.value import ValueNetwork
from synplan.utils.config import ReactorConfig
from synplan.utils.files import (
    count_sdf_records,
    count_smiles_records,
    iter_csv_smiles,
    iter_csv_smiles_blocks,
    iter_sdf_text_blocks,
    iter_smiles,
    iter_smiles_blocks,
)
from synplan.utils.parallel import process_pool_map_stream

if TYPE_CHECKING:
    from synplan.mcts.evaluation import (
        EvaluationStrategy,
        ValueNetworkEvaluationStrategy,
    )
    from synplan.mcts.policy import CompositePolicy, TemplateBasedPolicy
    from synplan.utils.config import (
        CombinedPolicyConfig,
        PolicyNetworkConfig,
        ValueNetworkConfig,
    )

REPO_ID = "Laboratoire-De-Chemoinformatique/SynPlanner-data"
LEGACY_REPO_ID = "Laboratoire-De-Chemoinformatique/SynPlanner"
logger = logging.getLogger(__name__)


def _building_blocks_progress(total: int | None, *, silent: bool):
    """Create a consistent progress bar for building blocks loading."""
    if silent:
        return None
    return tqdm(
        total=total,
        desc="Building blocks",
        unit="mol",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
        smoothing=0.1,
        disable=silent,
    )


def _map_blocks(blocks, worker_fn, *, num_workers: int):
    """Map blocks through worker function, optionally using a process pool.

    For `num_workers == 1`, this runs sequentially to avoid process-spawn overhead.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_workers == 1:
        for block in blocks:
            yield worker_fn(block)
        return
    yield from process_pool_map_stream(blocks, worker_fn, max_workers=num_workers)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip into `out_dir` only if its contents are missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            target = out_dir / name
            if not target.exists():
                zf.extract(name, out_dir)


def download_selected_files(
    files_to_get: list[tuple[str, str]],
    save_to: str | Path = "./tutorials/synplan_data",
    extract_zips: bool = True,
    relocate_map: dict[str, str] | None = None,
    repo_id: str | None = None,
) -> Path:
    """
    Download specific files from the Hugging Face repo.

    Parameters
    ----------
    files_to_get : list of (subfolder, filename)
        Example: [("building_blocks", "building_blocks_em_sa_ln.smi.zip"),
                  ("uspto", "uspto_reaction_rules.pickle"),
                  ("weights", "ranking_policy_network.ckpt")]
    save_to : path
        Where to save everything locally.
    extract_zips : bool
        If True, extract .zip files to their containing folder.
    relocate_map : dict[str, str]
        Optional map { "weights/ranking_policy_network.ckpt": "uspto/weights/ranking_policy_network.ckpt" }
        to copy/move files after download to match test paths.
    repo_id : str or None
        Override the HuggingFace repo ID. Defaults to ``REPO_ID``.
    """
    repo = repo_id or REPO_ID
    root = Path(save_to).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for subfolder, filename in files_to_get:
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                subfolder=subfolder,
                filename=filename,
                local_dir=str(root),
            )
        )

        if extract_zips and local_path.suffix == ".zip":
            _extract_zip(local_path, local_path.parent)

    if relocate_map:
        for src_rel, dst_rel in relocate_map.items():
            src = root / src_rel
            dst = root / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    return root


def download_unpack_data(filename, subfolder, save_to=".", repo_id=None):
    repo = repo_id or REPO_ID
    if isinstance(save_to, str):
        save_to = Path(save_to).resolve()
        save_to.mkdir(exist_ok=True)

    # Download the file from the repository
    file_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        subfolder=subfolder,
        local_dir=save_to,
    )
    file_path = Path(file_path)

    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # Extract the single file in the zip
            zip_ref.extractall(save_to)
            extracted_file = save_to / zip_ref.namelist()[0]

        file_path.unlink()

        return extracted_file
    else:
        return file_path


def download_preset(
    preset_name: str = "synplanner-gps",
    save_to: str | Path = ".",
    repo_id: str | None = None,
) -> dict[str, Path]:
    """Download a ready-to-use data preset from HuggingFace.

    The preset YAML lists explicit file paths under a ``files:`` key.
    Each file is downloaded into the ``save_to`` directory, preserving
    the repository folder structure.

    :param preset_name: Name of the preset (e.g. ``"synplanner-gps"``).
    :param save_to: Local directory to save downloaded files.
    :param repo_id: Override the HuggingFace repo ID.
    :return: Dict mapping component keys to local file paths.
    """
    repo = repo_id or REPO_ID
    root = Path(save_to).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 1. Download and parse preset YAML
    preset_path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=f"{preset_name}.yaml",
            subfolder="presets",
            local_dir=str(root),
        )
    )
    with open(preset_path, encoding="utf-8") as f:
        preset = yaml.safe_load(f)

    # 2. Download each file listed in the preset
    result: dict[str, Path] = {}
    for key, repo_path in preset.get("files", {}).items():
        parts = PurePosixPath(repo_path)
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=parts.name,
                subfolder=str(parts.parent),
                local_dir=str(root),
            )
        )
        result[key] = local_path

    return result


def download_all_data(save_to="."):
    """Download all data from the legacy HuggingFace repo.

    .. deprecated::
        Use :func:`download_preset` instead.
    """
    warnings.warn(
        "download_all_data() is deprecated. Use download_preset() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    dir_path = snapshot_download(repo_id=LEGACY_REPO_ID, local_dir=save_to)
    dir_path = Path(dir_path).resolve()
    for zip_file in dir_path.rglob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            # Check each file in the zip
            for file_name in zip_ref.namelist():
                extracted_file_path = zip_file.parent / file_name

                # Check if the extracted file already exists
                if not extracted_file_path.exists():
                    # Extract the file if it does not exist
                    zip_ref.extract(file_name, zip_file.parent)
                    print(f"Extracted {file_name} to {zip_file.parent}")


@functools.cache
def load_reaction_rules(
    file: str,
    reactor_config: ReactorConfig | None = None,
    *,
    check_atom_mapping: "AtomMappingCheck" = "reject_unmapped",
) -> tuple[CanonicalRetroReactor, ...]:
    """Loads the reaction rules from a TSV or pickle file and converts them into a
    tuple of Reactor objects.

    Supported formats

    - ``.tsv`` -- tab-separated text with ``rule_smarts``, ``popularity``,
      ``reaction_indices`` columns (preferred).
    - ``.pickle`` -- legacy pickle format (deprecated).

    The result is a tuple so it can be safely shared across parallel tree
    workers without risk of accidental mutation. ``functools.cache`` returns
    the same object on subsequent calls, so immutability also protects the
    cached value.

    :param file: The path to the file that stores the reaction rules.
    :param reactor_config: Optional ReactorConfig to control Reactor construction
        (e.g. automorphism_filter, delete_atoms). If None, uses defaults.
    :param check_atom_mapping: atom-mapping inspection mode applied to each
        SMARTS row before ``Reactor.from_smarts``. ``"reject_unmapped"`` (the
        default) rejects fully unmapped rules and allows partials (legitimate
        leaving/incoming groups). Only honoured for the TSV path; pickled
        rules are pre-compiled Reactor objects that can't be string-checked.
    :return: A tuple of reaction rules as Reactor objects.
    """
    ext = Path(file).suffix.lower()

    reactor_kwargs = reactor_config.to_reactor_kwargs() if reactor_config else {}

    if ext == ".tsv":
        return _load_rules_tsv(
            file, reactor_kwargs, check_atom_mapping=check_atom_mapping
        )

    # Legacy pickle path — cannot string-check pre-compiled Reactors.
    return _load_rules_pickle(file)


def _load_rules_tsv(
    file: str,
    reactor_kwargs: dict | None = None,
    *,
    check_atom_mapping: "AtomMappingCheck" = "reject_unmapped",
) -> tuple[CanonicalRetroReactor, ...]:
    """Load reaction rules from a TSV file."""
    if reactor_kwargs is None:
        reactor_kwargs = {}
    reactor_kwargs.setdefault("delete_atoms", False)
    reactors: list[CanonicalRetroReactor] = []
    with open(file, encoding="utf-8") as f:
        f.readline()  # skip header
        for row_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            smarts_str = parts[0]
            if check_atom_mapping != "off":
                status = reaction_string_mapping_status(smarts_str)
                if status == "unmapped" or (
                    status == "partially_mapped"
                    and check_atom_mapping == "reject_partial"
                ):
                    raise ValueError(
                        f"Rule at row {row_num} of {file!r} is {status}:\n"
                        f"  SMARTS: {smarts_str}\n"
                        "  set check_atom_mapping='off' to load it anyway."
                    )
            try:
                reactors.append(
                    CanonicalRetroReactor.from_smarts(smarts_str, **reactor_kwargs)
                )
            except Exception as err:
                raise ValueError(
                    f"Failed to load reaction rule at row {row_num} of "
                    f"{file!r}:\n  SMARTS: {smarts_str}\n"
                    f"  error: {type(err).__name__}: {err}"
                ) from err
    return tuple(reactors)


def _load_rules_pickle(file: str) -> tuple[CanonicalRetroReactor, ...]:
    """Load reaction rules from a legacy pickle file."""
    with open(file, "rb") as f:
        reaction_rules = pickle.load(f)

    # Already a list of chython Reactors (converted pickle)
    if isinstance(reaction_rules[0], Reactor):
        return tuple(reaction_rules)

    # Legacy format: list of (rule, priority) tuples; unpack to bare Reactors.
    if isinstance(reaction_rules[0][0], Reactor):
        reaction_rules = [rule for rule, _ in reaction_rules]

    return tuple(reaction_rules)


@functools.cache
def load_building_blocks(
    building_blocks_path: str | Path,
    standardize: bool = True,
    silent: bool = True,
    num_workers: int | None = None,
    chunksize: int = 1000,
    *,
    header: bool = True,
    delimiter: str = ",",
    smiles_column: str = "SMILES",
) -> frozenset[str]:
    """Loads building blocks data from a file and returns a frozen set of building
    blocks.

    :param building_blocks_path: The path to the file containing the building blocks.
    :param standardize: Flag if building blocks have to be standardized before loading. Default=True.
    :param header: For CSV/CSV.GZ files: treat the first row as header. Default=True.
    :param delimiter: For CSV/CSV.GZ files: delimiter character. Default=",".
    :param smiles_column: For CSV/CSV.GZ files: header column name containing SMILES.
        Default="SMILES" (case-insensitive match is supported).
    :return: The set of building blocks smiles.
    """

    building_blocks_path = Path(building_blocks_path).resolve()
    suffixes = "".join(building_blocks_path.suffixes).lower()
    is_csv = suffixes.endswith(".csv") or suffixes.endswith(".csv.gz")
    is_tsv = suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz")
    if is_tsv:
        is_csv = True
        delimiter = "\t"
    suffix = building_blocks_path.suffix.lower()
    if not is_csv and suffix not in {".smi", ".smiles", ".sdf"}:
        raise ValueError(
            f"Unsupported building blocks file extension: '{building_blocks_path.name}'. "
            "Supported: .smi, .smiles, .sdf, .csv, .csv.gz, .tsv, .tsv.gz"
        )

    building_blocks_smiles = set()
    if standardize:
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        if suffix in {".smi", ".smiles"}:
            total = count_smiles_records(building_blocks_path) if not silent else None
            step = max(1, chunksize or 1000)

            progress_iter = _building_blocks_progress(total, silent=silent)
            for out in _map_blocks(
                iter_smiles_blocks(building_blocks_path, step),
                standardize_smiles_batch,
                num_workers=num_workers,
            ):
                if out:
                    building_blocks_smiles.update(out)
                    if progress_iter is not None:
                        progress_iter.update(len(out))
            if progress_iter is not None:
                progress_iter.close()

        elif is_csv:
            step = max(1, chunksize or 1000)
            progress_iter = _building_blocks_progress(None, silent=silent)
            blocks = iter_csv_smiles_blocks(
                building_blocks_path,
                step,
                header=header,
                delimiter=delimiter,
                smiles_column=smiles_column,
            )
            for out in _map_blocks(
                blocks, standardize_smiles_batch, num_workers=num_workers
            ):
                if out:
                    building_blocks_smiles.update(out)
                    if progress_iter is not None:
                        progress_iter.update(len(out))
            if progress_iter is not None:
                progress_iter.close()

        elif suffix == ".sdf":
            n = count_sdf_records(building_blocks_path) if not silent else None
            step = max(1, chunksize or 5000)
            blocks = iter_sdf_text_blocks(building_blocks_path, step)

            progress = _building_blocks_progress(n, silent=silent)
            for chunk_out in _map_blocks(
                blocks, standardize_sdf_text, num_workers=num_workers
            ):
                if chunk_out:
                    building_blocks_smiles.update(chunk_out)
                    if progress is not None:
                        progress.update(len(chunk_out))
            if progress is not None:
                progress.close()
    else:
        if suffix in {".smi", ".smiles"}:
            for smiles in iter_smiles(building_blocks_path):
                building_blocks_smiles.add(smiles)
        elif is_csv:
            for smiles in iter_csv_smiles(
                building_blocks_path,
                header=header,
                delimiter=delimiter,
                smiles_column=smiles_column,
            ):
                building_blocks_smiles.add(smiles)
        elif suffix == ".sdf":
            with SDFRead(str(building_blocks_path)) as sdf:
                for mol in sdf:
                    with contextlib.suppress(Exception):
                        building_blocks_smiles.add(str(mol))

    return frozenset(building_blocks_smiles)


def load_value_net(
    model_class: type[ValueNetwork], value_network_path: str | Path
) -> ValueNetwork:
    """Loads the value network.

    :param value_network_path: The path to the file storing value network weights.
    :param model_class: The model class to be loaded.
    :return: The loaded value network.
    """
    from synplan.ml.networks.checkpoint import load_network_from_checkpoint

    return load_network_from_checkpoint(
        model_class, value_network_path, map_location="cpu"
    )


def build_policy_from_config(
    policy_config: "PolicyNetworkConfig",
) -> "TemplateBasedPolicy":
    """Build a :class:`TemplateBasedPolicy` matching a checkpoint architecture."""
    from synplan.mcts.policy import LinearPolicy, MHNReactPolicy
    from synplan.ml.networks.checkpoint import load_policy_network_from_checkpoint

    policy_net = load_policy_network_from_checkpoint(
        policy_config.weights_path, batch_size=1, dropout=0
    )
    knobs = dict(
        top_rules=policy_config.top_rules,
        rule_prob_threshold=policy_config.rule_prob_threshold,
        priority_rules_fraction=policy_config.priority_rules_fraction,
    )
    if getattr(policy_net, "architecture", "linear") == "mhn_ranking":
        return MHNReactPolicy(policy_net, **knobs)
    return LinearPolicy(policy_net, **knobs)


def load_policy_function(
    policy_config: Union["PolicyNetworkConfig", dict, None] = None,
    weights_path: str | None = None,
    **config_kwargs,
) -> "TemplateBasedPolicy":
    """Build a template-based :class:`Policy` from flexible configuration.

    Priority order: policy_config > weights_path + kwargs > defaults

    :param policy_config: PolicyNetworkConfig object or dict with config parameters
    :param weights_path: Direct path to weights file (shortcut for simple cases)
    :param config_kwargs: Additional config parameters to override defaults
    :return: A :class:`TemplateBasedPolicy` ready for use in tree search

    Examples:
        >>> # Using config object
        >>> config = PolicyNetworkConfig(weights_path="path.ckpt", top_rules=50)
        >>> policy = load_policy_function(policy_config=config)
        >>>
        >>> # Using direct path (simplest)
        >>> policy = load_policy_function(weights_path="path.ckpt")
        >>>
        >>> # Using path with overrides
        >>> policy = load_policy_function(weights_path="path.ckpt", top_rules=100)
    """
    from synplan.utils.config import PolicyNetworkConfig

    if policy_config is not None:
        if isinstance(policy_config, dict):
            policy_config = PolicyNetworkConfig.from_dict(policy_config)
        return build_policy_from_config(policy_config)

    if weights_path is not None:
        policy_config = PolicyNetworkConfig(weights_path=weights_path, **config_kwargs)
        return build_policy_from_config(policy_config)

    raise ValueError("Must provide either policy_config or weights_path")


def load_combined_policy_function(
    combined_config: Union["CombinedPolicyConfig", dict] = None,
    filtering_config: Union["PolicyNetworkConfig", dict, str] = None,
    ranking_config: Union["PolicyNetworkConfig", dict, str] = None,
    filtering_weights_path: str | None = None,
    ranking_weights_path: str | None = None,
    top_rules: int = 50,
    rule_prob_threshold: float = 0.0,
    ranking_weight: float = 1.0,
    temperature: float = 1.0,
) -> "CompositePolicy":
    """Build a :class:`CompositePolicy` merging filtering and ranking policies.

    Combines filtering and ranking policies by weighted addition of logits:
        combined_logits = filtering_logits + ranking_weight * ranking_logits
        combined_probs = softmax(combined_logits / temperature)

    The filtering policy provides applicability scores (trained on multi-label applicability).
    The ranking policy provides feasibility scores (trained on actual reactions).

    :param combined_config: CombinedPolicyConfig or dict with all parameters.
    :param filtering_config: PolicyNetworkConfig or dict for filtering policy.
    :param ranking_config: PolicyNetworkConfig or dict for ranking policy.
    :param filtering_weights_path: Direct path to filtering weights (shortcut).
    :param ranking_weights_path: Direct path to ranking weights (shortcut).
    :param top_rules: Number of top rules to return.
    :param rule_prob_threshold: Minimum probability threshold for returning a rule.
    :param ranking_weight: Weight for ranking logits (default 1.0).
        Values > 1.0 give more weight to ranking (feasibility).
    :param temperature: Temperature for softmax (default 1.0).
        Values > 1.0 produce softer distributions (more exploration).
    :return: A :class:`CompositePolicy` ready for use in tree search.

    Examples:
        >>> # Using CombinedPolicyConfig
        >>> config = CombinedPolicyConfig(
        ...     filtering_weights_path="filtering.ckpt",
        ...     ranking_weights_path="ranking.ckpt",
        ... )
        >>> combined = load_combined_policy_function(combined_config=config)
        >>>
        >>> # Using config objects
        >>> combined = load_combined_policy_function(
        ...     filtering_config={"weights_path": "filtering.ckpt", "policy_type": "filtering"},
        ...     ranking_config={"weights_path": "ranking.ckpt", "policy_type": "ranking"},
        ... )
        >>>
        >>> # Using direct paths (simplest)
        >>> combined = load_combined_policy_function(
        ...     filtering_weights_path="filtering.ckpt",
        ...     ranking_weights_path="ranking.ckpt",
        ... )
    """
    from synplan.mcts.policy import CompositePolicy
    from synplan.utils.config import CombinedPolicyConfig, PolicyNetworkConfig

    # Priority 1: Use CombinedPolicyConfig
    if combined_config is not None:
        if isinstance(combined_config, dict):
            combined_config = CombinedPolicyConfig.from_dict(combined_config)
        filtering_weights_path = combined_config.filtering_weights_path
        ranking_weights_path = combined_config.ranking_weights_path
        top_rules = combined_config.top_rules
        rule_prob_threshold = combined_config.rule_prob_threshold
        ranking_weight = combined_config.ranking_weight
        temperature = combined_config.temperature
        filtering_config = PolicyNetworkConfig(
            weights_path=filtering_weights_path, policy_type="filtering"
        )
        ranking_config = PolicyNetworkConfig(
            weights_path=ranking_weights_path, policy_type="ranking"
        )
    else:
        # Build filtering config
        if filtering_config is not None:
            if isinstance(filtering_config, str):
                filtering_config = PolicyNetworkConfig(
                    weights_path=filtering_config, policy_type="filtering"
                )
            elif isinstance(filtering_config, dict):
                filtering_config.setdefault("policy_type", "filtering")
                filtering_config = PolicyNetworkConfig.from_dict(filtering_config)
        elif filtering_weights_path is not None:
            filtering_config = PolicyNetworkConfig(
                weights_path=filtering_weights_path, policy_type="filtering"
            )
        else:
            raise ValueError(
                "Must provide either filtering_config or filtering_weights_path"
            )

        # Build ranking config
        if ranking_config is not None:
            if isinstance(ranking_config, str):
                ranking_config = PolicyNetworkConfig(
                    weights_path=ranking_config, policy_type="ranking"
                )
            elif isinstance(ranking_config, dict):
                ranking_config.setdefault("policy_type", "ranking")
                ranking_config = PolicyNetworkConfig.from_dict(ranking_config)
        elif ranking_weights_path is not None:
            ranking_config = PolicyNetworkConfig(
                weights_path=ranking_weights_path, policy_type="ranking"
            )
        else:
            raise ValueError(
                "Must provide either ranking_config or ranking_weights_path"
            )

    if filtering_config.policy_type != "filtering":
        raise ValueError(
            f"filtering_config must have policy_type='filtering', got "
            f"'{filtering_config.policy_type}'"
        )
    if ranking_config.policy_type != "ranking":
        raise ValueError(
            f"ranking_config must have policy_type='ranking', got "
            f"'{ranking_config.policy_type}'"
        )

    return CompositePolicy(
        build_policy_from_config(filtering_config),
        build_policy_from_config(ranking_config),
        top_rules=top_rules,
        rule_prob_threshold=rule_prob_threshold,
        ranking_weight=ranking_weight,
        temperature=temperature,
    )


def load_value_network(
    value_config: Union["ValueNetworkConfig", dict, None] = None,
    weights_path: str | None = None,
    normalize: bool = False,
    **config_kwargs,
) -> "ValueNetworkEvaluationStrategy":
    """Factory function to create a value evaluation strategy.

    Priority order: value_config > weights_path + kwargs > defaults

    :param value_config: ValueNetworkConfig object or dict with config parameters
    :param weights_path: Direct path to weights file (shortcut for simple cases)
    :param normalize: Whether to normalize scores to [0, 1].
    :param config_kwargs: Additional config parameters to override defaults
    :return: ValueNetworkEvaluationStrategy ready for use in tree search

    Examples:
        >>> # Using config object
        >>> config = ValueNetworkConfig(weights_path="path.ckpt")
        >>> value_fn = load_value_network(value_config=config)
        >>>
        >>> # Using direct path (simplest)
        >>> value_fn = load_value_network(weights_path="path.ckpt")
    """
    from synplan.mcts.evaluation import ValueNetworkEvaluationStrategy
    from synplan.utils.config import ValueNetworkConfig

    if value_config is not None:
        if isinstance(value_config, dict):
            value_config = ValueNetworkConfig.from_dict(value_config)
        weights_path = value_config.weights_path
    if weights_path is None:
        raise ValueError("Must provide either value_config or weights_path")

    return ValueNetworkEvaluationStrategy(
        weights_path=weights_path, normalize=normalize
    )


def load_evaluation_function(eval_config) -> "EvaluationStrategy":
    """Create evaluation strategy from configuration.

    This is the central factory function that creates the appropriate evaluation
    strategy based on the config type. The config contains all necessary dependencies.

    :param eval_config: Evaluation configuration object (self-contained).
        Can be one of:
        - RolloutEvaluationConfig
        - ValueNetworkEvaluationConfig
        - RDKitEvaluationConfig
        - PolicyEvaluationConfig
        - RandomEvaluationConfig
    :return: Evaluation strategy ready to use in tree search.

    Examples:
        >>> # Rollout evaluation
        >>> config = RolloutEvaluationConfig(
        ...     policy_network=policy,
        ...     reaction_rules=rules,
        ...     building_blocks=bbs,
        ...     max_depth=9
        ... )
        >>> evaluator = load_evaluation_function(config)
        >>>
        >>> # Value network evaluation
        >>> config = ValueNetworkEvaluationConfig(weights_path="path.ckpt")
        >>> evaluator = load_evaluation_function(config)
    """
    from synplan.mcts.evaluation import (
        PolicyEvaluationStrategy,
        RandomEvaluationStrategy,
        RDKitEvaluationStrategy,
        RolloutEvaluationStrategy,
        ValueNetworkEvaluationStrategy,
    )
    from synplan.utils.config import (
        PolicyEvaluationConfig,
        RandomEvaluationConfig,
        RDKitEvaluationConfig,
        RolloutEvaluationConfig,
        ValueNetworkEvaluationConfig,
    )

    logger.debug(f"create_evaluator config_type={type(eval_config).__name__}")
    if isinstance(eval_config, RolloutEvaluationConfig):
        return RolloutEvaluationStrategy(
            policy_network=eval_config.policy_network,
            reaction_rules=eval_config.reaction_rules,
            building_blocks=eval_config.building_blocks,
            min_mol_size=eval_config.min_mol_size,
            max_depth=eval_config.max_depth,
            normalize=eval_config.normalize,
            stochastic=eval_config.stochastic,
        )

    elif isinstance(eval_config, ValueNetworkEvaluationConfig):
        return ValueNetworkEvaluationStrategy(
            weights_path=eval_config.weights_path,
            normalize=eval_config.normalize,
        )

    elif isinstance(eval_config, RDKitEvaluationConfig):
        return RDKitEvaluationStrategy(
            score_function=eval_config.score_function,
            normalize=eval_config.normalize,
        )

    elif isinstance(eval_config, PolicyEvaluationConfig):
        return PolicyEvaluationStrategy(normalize=eval_config.normalize)

    elif isinstance(eval_config, RandomEvaluationConfig):
        return RandomEvaluationStrategy(normalize=eval_config.normalize)

    else:
        raise ValueError(
            f"Unknown evaluation config type: {type(eval_config)}. "
            f"Expected one of: RolloutEvaluationConfig, ValueNetworkEvaluationConfig, "
            f"RDKitEvaluationConfig, PolicyEvaluationConfig, RandomEvaluationConfig."
        )
