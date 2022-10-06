import json
import logging
import zipfile
from pathlib import Path

import requests

import yuca.config as cfg
from yuca.dataset import Dataset

SUPPORTED_DATASETS = [
    "geolife",
    "mnist",
    "langevin",
    "diff_diff",
    "langevin_vs_diff_diff",
]






def _version_regeneration(old_version: str, new_version: str) -> bool:
    """Checks if the change in version requires a regeneration."""
    old_minor = old_version.split(".")[1]
    new_minor = new_version.split(".")[1]
    return old_minor != new_minor


def _load_geolife_dataset(force_redownload: bool) -> Dataset:
    dataset_name = "geolife"
    # Check if the dataset needs to be downloaded
    if _needs_download(dataset_name, force_redownload):
        _download_dataset(cfg.GEOLIFE_DS_URL, dataset_name)

    # Load the dataset metadata
    metadata = _load_metadata(_get_path(cfg.DS_METADATA_FILE, dataset_name))

    if _version_regeneration(metadata["version"], cfg.DS_VERSION):
        # Regenerate the yupi data
        raise NotImplementedError

    # Load the yupi data
    raise NotImplementedError


def _load_mnist_dataset(force_redownload: bool) -> Dataset:
    # Check if the dataset needs to be downloaded
    # parse and yupify the dataset
    raise NotImplementedError


def _load_langevin_dataset() -> Dataset:
    # Generate trajectories
    # Build dataset
    raise NotImplementedError


def _load_diff_diff_dataset() -> Dataset:
    # Generate trajectories
    # Build dataset
    raise NotImplementedError


def _load_langevin_vs_diff_diff_dataset() -> Dataset:
    # Generate trajectories
    # Build dataset
    raise NotImplementedError


def load_dataset(datase_name: str, force_redownload: bool = False) -> Dataset:
    """
    Loads a dataset from the datasets folder.

    Parameters
    ----------
    datase_name : str
        The name of the dataset. Must be one of the supported datasets.
    force_redownload : bool, optional
        If True, the dataset will be downloaded again, even if it already exists.

        Only applies to the 'geolife' and 'mnist' datasets.
    """
    if datase_name == "geolife":
        return _load_geolife_dataset(force_redownload)
    if datase_name == "mnist":
        return _load_mnist_dataset(force_redownload)
    if datase_name == "langevin":
        return _load_langevin_dataset()
    if datase_name == "diff_diff":
        return _load_diff_diff_dataset()
    if datase_name == "langevin_vs_diff_diff":
        return _load_langevin_vs_diff_diff_dataset()
    raise ValueError(
        f"The dataset {datase_name} is not supported.\n"
        "Supported datasets are:\n"
        f"{', '.join(SUPPORTED_DATASETS)}"
    )
