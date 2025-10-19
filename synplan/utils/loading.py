"""Module containing functions for loading reaction rules, building blocks and
retrosynthetic models."""

import functools
import pickle
import zipfile
import shutil
from pathlib import Path
from typing import List, Set, FrozenSet, Tuple, Union

from CGRtools.reactor.reactor import Reactor
from CGRtools.files.SDFrw import SDFRead
from torch import device
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

from synplan.ml.networks.policy import PolicyNetwork
from synplan.ml.networks.value import ValueNetwork
from synplan.utils.files import MoleculeReader
from synplan.chem import smiles_parser
from synplan.chem.utils import (
    safe_canonicalization,
    _standardize_one_smiles,
    _standardize_sdf_range,
)

REPO_ID = "Laboratoire-De-Chemoinformatique/SynPlanner"


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
    if not standardize:
        assert suffix in {".smi", ".smiles"}
    else:
        assert suffix in {".smi", ".smiles", ".sdf"}

    building_blocks_smiles = set()
    if standardize:
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)

        if suffix in {".smi", ".smiles"}:

            def _line_iter():
                with open(building_blocks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield line.split()[0]

            total = None
            if not silent:
                with open(building_blocks_path, "r", encoding="utf-8") as f:
                    total = sum(1 for _ in f)

            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                results = ex.map(
                    _standardize_one_smiles, _line_iter(), chunksize=chunksize or 1000
                )
                if not silent:
                    results = tqdm(
                        results,
                        total=total,
                        desc="Number of building blocks processed: ",
                        bar_format="{desc}{n} [{elapsed}]",
                        disable=silent,
                    )
                for res in results:
                    if res:
                        building_blocks_smiles.add(res)

        elif suffix == ".sdf":
            sdf = SDFRead(str(building_blocks_path), indexable=True)
            n = len(sdf)
            sdf.close()

            step = max(1, chunksize or 5000)
            ranges = [
                (str(building_blocks_path), i, min(i + step, n))
                for i in range(0, n, step)
            ]

            progress = None
            if not silent:
                progress = tqdm(
                    total=n,
                    desc="Number of building blocks processed: ",
                    bar_format="{desc}{n} [{elapsed}]",
                    disable=silent,
                )

            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for chunk_out in ex.map(
                    lambda args: _standardize_sdf_range(*args), ranges
                ):
                    if chunk_out:
                        building_blocks_smiles.update(chunk_out)
                        if progress is not None:
                            progress.update(len(chunk_out))
            if progress is not None:
                progress.close()
    else:
        with open(building_blocks_path, "r") as inp:
            for line in inp:
                smiles = line.strip().split()[0]
                building_blocks_smiles.add(smiles)

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
