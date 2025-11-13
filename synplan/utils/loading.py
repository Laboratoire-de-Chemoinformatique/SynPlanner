"""Module containing functions for loading reaction rules, building blocks and
retrosynthetic models."""

import functools
import logging
import os
from pathlib import Path
import pickle
import shutil
from typing import FrozenSet, List, TYPE_CHECKING, Union
import zipfile

from CGRtools.files.SDFrw import SDFRead
from CGRtools.reactor.reactor import Reactor
from huggingface_hub import hf_hub_download, snapshot_download
from torch import device
from tqdm.auto import tqdm

from synplan.chem.utils import _standardize_sdf_text, _standardize_smiles_batch
from synplan.ml.networks.policy import PolicyNetwork
from synplan.ml.networks.value import ValueNetwork
from synplan.utils.files import (
    count_sdf_records,
    count_smiles_records,
    iter_sdf_text_blocks,
    iter_smiles,
    iter_smiles_blocks,
)
from synplan.utils.parallel import process_pool_map_stream

if TYPE_CHECKING:
    from synplan.utils.config import ValueNetworkConfig, PolicyNetworkConfig
    from synplan.mcts.expansion import PolicyNetworkFunction
    from synplan.mcts.evaluation import ValueNetworkFunction, EvaluationStrategy

REPO_ID = "Laboratoire-De-Chemoinformatique/SynPlanner"
logger = logging.getLogger(__name__)


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
    """
    root = Path(save_to).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for subfolder, filename in files_to_get:
        local_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
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
                shutil.copy2(src, dst)  # or shutil.move(src, dst)

    return root


def download_unpack_data(filename, subfolder, save_to="."):
    if isinstance(save_to, str):
        save_to = Path(save_to).resolve()
        save_to.mkdir(exist_ok=True)

    # Download the zip file from the repository
    file_path = hf_hub_download(
        repo_id=REPO_ID,
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


def download_all_data(save_to="."):
    dir_path = snapshot_download(repo_id=REPO_ID, local_dir=save_to)
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


@functools.lru_cache(maxsize=None)
def load_reaction_rules(file: str) -> List[Reactor]:
    """Loads the reaction rules from a pickle file and converts them into a list of
    Reactor objects if necessary.

    :param file: The path to the pickle file that stores the reaction rules.
    :return: A list of reaction rules as Reactor objects.
    """

    with open(file, "rb") as f:
        reaction_rules = pickle.load(f)

    if not isinstance(reaction_rules[0][0], Reactor):
        reaction_rules = [Reactor(x) for x, _ in reaction_rules]

    return tuple(reaction_rules)


@functools.lru_cache(maxsize=None)
def load_building_blocks(
    building_blocks_path: Union[str, Path],
    standardize: bool = True,
    silent: bool = True,
    num_workers: int | None = None,
    chunksize: int = 1000,
) -> FrozenSet[str]:
    """Loads building blocks data from a file and returns a frozen set of building
    blocks.

    :param building_blocks_path: The path to the file containing the building blocks.
    :param standardize: Flag if building blocks have to be standardized before loading. Default=True.
    :return: The set of building blocks smiles.
    """

    building_blocks_path = Path(building_blocks_path).resolve()
    suffix = building_blocks_path.suffix.lower()
    assert suffix in {".smi", ".smiles", ".sdf"}

    building_blocks_smiles = set()
    if standardize:
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)

        if suffix in {".smi", ".smiles"}:

            total = count_smiles_records(building_blocks_path) if not silent else None
            step = max(1, chunksize or 1000)

            if not silent:
                progress_iter = tqdm(
                    total=total,
                    desc="Building blocks",
                    unit="mol",
                    unit_scale=True,
                    unit_divisor=1000,
                    dynamic_ncols=True,
                    smoothing=0.1,
                    disable=silent,
                )
            else:
                progress_iter = None

            for out in process_pool_map_stream(
                iter_smiles_blocks(building_blocks_path, step),
                _standardize_smiles_batch,
                max_workers=num_workers,
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

            progress = None
            if not silent:
                progress = tqdm(
                    total=n,
                    desc="Building blocks",
                    unit="mol",
                    unit_scale=True,
                    unit_divisor=1000,
                    dynamic_ncols=True,
                    smoothing=0.1,
                    disable=silent,
                )

            for chunk_out in process_pool_map_stream(
                blocks, _standardize_sdf_text, max_workers=num_workers
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
        elif suffix == ".sdf":
            with SDFRead(str(building_blocks_path)) as sdf:
                for mol in sdf:
                    try:
                        building_blocks_smiles.add(str(mol))
                    except Exception:
                        pass

    return frozenset(building_blocks_smiles)


def load_value_net(
    model_class: ValueNetwork, value_network_path: Union[str, Path]
) -> ValueNetwork:
    """Loads the value network.

    :param value_network_path: The path to the file storing value network weights.
    :param model_class: The model class to be loaded.
    :return: The loaded value network.
    """

    map_location = device("cpu")
    return model_class.load_from_checkpoint(value_network_path, map_location)


def load_policy_net(
    model_class: PolicyNetwork, policy_network_path: Union[str, Path]
) -> PolicyNetwork:
    """Loads the policy network.

    :param policy_network_path: The path to the file storing policy network weights.
    :param model_class: The model class to be loaded.
    :return: The loaded policy network.
    """

    map_location = device("cpu")
    return model_class.load_from_checkpoint(
        policy_network_path, map_location, batch_size=1
    )


def load_policy_function(
    policy_config: Union["PolicyNetworkConfig", dict, None] = None,
    weights_path: str = None,
    **config_kwargs,
) -> "PolicyNetworkFunction":
    """Factory function to create PolicyNetworkFunction with flexible configuration.

    Priority order: policy_config > weights_path + kwargs > defaults

    :param policy_config: PolicyNetworkConfig object or dict with config parameters
    :param weights_path: Direct path to weights file (shortcut for simple cases)
    :param config_kwargs: Additional config parameters to override defaults
    :return: PolicyNetworkFunction ready for use in tree search

    Examples:
        >>> # Using config object
        >>> config = PolicyNetworkConfig(weights_path="path.ckpt", top_rules=50)
        >>> policy_fn = load_policy_function(policy_config=config)
        >>>
        >>> # Using direct path (simplest)
        >>> policy_fn = load_policy_function(weights_path="path.ckpt")
        >>>
        >>> # Using path with overrides
        >>> policy_fn = load_policy_function(weights_path="path.ckpt", top_rules=100)
    """
    from synplan.mcts.expansion import PolicyNetworkFunction
    from synplan.utils.config import PolicyNetworkConfig

    # Priority 1: Use provided config
    if policy_config is not None:
        if isinstance(policy_config, dict):
            policy_config = PolicyNetworkConfig.from_dict(policy_config)
        return PolicyNetworkFunction(policy_config=policy_config)

    # Priority 2: Create config from weights_path and kwargs
    if weights_path is not None:
        policy_config = PolicyNetworkConfig(weights_path=weights_path)
        return PolicyNetworkFunction(policy_config=policy_config)

    raise ValueError("Must provide either policy_config or weights_path")


def load_value_network(
    value_config: Union["ValueNetworkConfig", dict, None] = None,
    weights_path: str = None,
    **config_kwargs,
) -> "ValueNetworkFunction":
    """Factory function to create ValueNetworkFunction with flexible configuration.

    Priority order: value_config > weights_path + kwargs > defaults

    :param value_config: ValueNetworkConfig object or dict with config parameters
    :param weights_path: Direct path to weights file (shortcut for simple cases)
    :param config_kwargs: Additional config parameters to override defaults
    :return: ValueNetworkFunction ready for use in tree search

    Examples:
        >>> # Using config object
        >>> config = ValueNetworkConfig(weights_path="path.ckpt")
        >>> value_fn = load_value_network(value_config=config)
        >>>
        >>> # Using direct path (simplest)
        >>> value_fn = load_value_network(weights_path="path.ckpt")
    """
    from synplan.mcts.evaluation import ValueNetworkFunction
    from synplan.utils.config import ValueNetworkConfig

    # Priority 1: Use provided config
    if value_config is not None:
        if isinstance(value_config, dict):
            value_config = ValueNetworkConfig.from_dict(value_config)
        # ValueNetworkFunction only takes weights_path
        return ValueNetworkFunction(weights_path=value_config.weights_path)

    # Priority 2: Use direct weights_path
    if weights_path is not None:
        return ValueNetworkFunction(weights_path=weights_path)

    raise ValueError("Must provide either value_config or weights_path")


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
        RolloutEvaluationStrategy,
        ValueNetworkEvaluationStrategy,
        RDKitEvaluationStrategy,
        PolicyEvaluationStrategy,
        RandomEvaluationStrategy,
    )
    from synplan.utils.config import (
        RolloutEvaluationConfig,
        ValueNetworkEvaluationConfig,
        RDKitEvaluationConfig,
        PolicyEvaluationConfig,
        RandomEvaluationConfig,
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
        )

    elif isinstance(eval_config, ValueNetworkEvaluationConfig):
        # Load value network from path in config
        value_net = load_value_network(weights_path=eval_config.weights_path)
        return ValueNetworkEvaluationStrategy(
            value_network=value_net,
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
