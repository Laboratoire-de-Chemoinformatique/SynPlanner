"""Module containing functions for loading reaction rules, building blocks and
retrosynthetic models."""

import functools
import pickle
import zipfile
from pathlib import Path
from typing import List, Set, Union

from CGRtools.reactor.reactor import Reactor
from torch import device
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from synplan.ml.networks.policy import PolicyNetwork
from synplan.ml.networks.value import ValueNetwork
from synplan.utils.files import MoleculeReader


def download_unpack_data(filename, subfolder, save_to="."):
    if isinstance(save_to, str):
        save_to = Path(save_to).resolve()
        save_to.mkdir(exist_ok=True)

    # Download the zip file from the repository
    file_path = hf_hub_download(
        repo_id="Laboratoire-De-Chemoinformatique/SynPlanner",
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
    dir_path = snapshot_download(
        repo_id="Laboratoire-De-Chemoinformatique/SynPlanner", local_dir=save_to
    )
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

    return reaction_rules


@functools.lru_cache(maxsize=None)
def load_building_blocks(
    building_blocks_path: Union[str, Path], standardize: bool = True
) -> Set[str]:
    """Loads building blocks data from a file and returns a frozen set of building
    blocks.

    :param building_blocks_path: The path to the file containing the building blocks.
    :param standardize: Flag if building blocks have to be standardized before loading. Default=True.
    :return: The set of building blocks smiles.
    """

    building_blocks_path = Path(building_blocks_path).resolve()
    assert (
        building_blocks_path.suffix == ".smi"
        or building_blocks_path.suffix == ".smiles"
    )

    building_blocks_smiles = set()
    if standardize:
        with MoleculeReader(building_blocks_path) as molecules:
            for mol in tqdm(
                molecules,
                desc="Number of building blocks processed: ",
                bar_format="{desc}{n} [{elapsed}]",
            ):
                try:
                    mol.canonicalize()
                    mol.clean_stereo()
                    building_blocks_smiles.add(str(mol))
                except:  # mol.canonicalize() / InvalidAromaticRing
                    pass
    else:
        with open(building_blocks_path, "r") as inp:
            for line in inp:
                smiles = line.strip().split()[0]
                building_blocks_smiles.add(smiles)

    return building_blocks_smiles


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
