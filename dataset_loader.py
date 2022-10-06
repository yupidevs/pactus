import json
import logging
import zipfile
from pathlib import Path

import requests

import config as cfg
from dataset import Dataset

SUPPORTED_DATASETS = [
    "geolife",
    "mnist",
    "langevin",
    "diff_diff",
    "langevin_vs_diff_diff",
]


def _get_path(path: str, *args) -> Path:
    return Path(path.format(*args))


def _load_metadata(path: Path) -> dict:
    assert path.exists(), f"Metadata file {path} does not exist"
    with open(path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _version_redownload(old_version: str, new_version: str) -> bool:
    """Checks if the change in version requires a redownload."""
    old_major = old_version.split(".")[0]
    new_major = new_version.split(".")[0]
    return old_major != new_major

def _version_regeneration(old_version: str, new_version: str) -> bool:
    """Checks if the change in version requires a regeneration."""
    old_minor = old_version.split(".")[1]
    new_minor = new_version.split(".")[1]
    return old_minor != new_minor


def _needs_download(dataset_name: str, force_redownload: bool = False) -> bool:
    """Checks if a dataset needs to be downloaded."""
    if force_redownload:
        return True

    metadata_file = _get_path(cfg.DS_METADATA_FILE, dataset_name)

    # If the metadata file doesn't exist, the dataset needs to be downloaded
    if not metadata_file.exists():
        return True

    metadata = _load_metadata(metadata_file)

    # Redownload if the version changes
    return _version_redownload(metadata["version"], cfg.DS_VERSION)


def _download_dataset(url: str, dataset_name: str) -> None:
    """Downloads a dataset from a url."""

    def get_progress_log(part, total):
        # If the total is unknown, just return the part
        if total == -1:
            return f"Downloaded: {part / 1024 ** 2:.2f} MB"

        passed = "#" * int(cfg.PROGRESS_BAR_LENGTH * part / total)
        rest = " " * (cfg.PROGRESS_BAR_LENGTH - len(passed))
        return f"[{passed}{rest}] {part * 100/total:.2f}%"

    # Create the dataset folder if it doesn't exist
    logging.info("Creating dataset folder for %s", dataset_name)
    dataset_path = _get_path(cfg.DS_DIR, dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Make the download request
    logging.info("Downloading %s dataset", dataset_name)
    response = requests.get(url, allow_redirects=True, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset {dataset_name}")

    # Download the dataset to a zip file
    data_length = int(response.headers.get("content-length", -1))
    dataset_file_path = dataset_path / url.split("/")[-1]
    with open(dataset_file_path, "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=cfg.DOWNLOAD_CHUNCK_SIZE):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(get_progress_log(downloaded, data_length), end="\r")

    # Extract the dataset
    logging.info("Extracting %s dataset", dataset_name)
    with zipfile.ZipFile(dataset_file_path, "r") as zip_ref:
        zip_ref.extractall(str(dataset_path))

    # Create the dataset metadata file
    logging.info("Creating dataset metadata for %s", dataset_name)
    metadata_path = _get_path(cfg.DS_METADATA_FILE, dataset_name)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        metadata = {
            "name": dataset_name,
            "path": str(dataset_path),
            "version": cfg.DS_VERSION,
        }
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)


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
