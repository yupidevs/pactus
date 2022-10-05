import json
import logging
import zipfile
from pathlib import Path

import requests

from dataset import Dataset

__GEOLIFE_DATASET_URL = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
__MNIST_DATASET_URL = "http://yann.lecun.com/exdb/mnist/"
__DOWNLOAD_CHUNCK_SIZE = 4096


def _dataset_path(dataset_name: str) -> Path:
    return Path(f"datasets/{dataset_name}")


def _dataset_metadata_path(dataset_name: str) -> Path:
    return _dataset_path(dataset_name) / "dataset_metadata.json"


def _dataset_raw_data_path(dataset_name: str) -> Path:
    return _dataset_path(dataset_name) / "raw_data"


def _dataset_yupified_data_path(dataset_name: str) -> Path:
    return _dataset_path(dataset_name) / "yupified_data"


def _download_dataset(url: str, dataset_name: str) -> None:
    """Downloads a dataset from a url."""

    def get_progress_log(part, total):
        if total == -1:
            return f"Downloaded {part / __DOWNLOAD_CHUNCK_SIZE} bytes"

        passed = int(50 * part / total)
        rest = int(50 * (1 - part / total))
        return f"[{'#' * passed}{' ' * rest}] {part * 100/total:.2f}%"

    # Create the dataset folder if it doesn't exist
    logging.info("Creating dataset folder for %s", dataset_name)
    dataset_path = _dataset_raw_data_path(dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    logging.info("Downloading %s dataset", dataset_name)
    response = requests.get(url, allow_redirects=True, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset {dataset_name}")
    data_length = int(response.headers.get("content-length", -1))
    with open(dataset_path / "dataset.zip", "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=__DOWNLOAD_CHUNCK_SIZE):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(get_progress_log(downloaded, data_length), end="\r")

    # Extract the dataset
    logging.info("Extracting %s dataset", dataset_name)
    with zipfile.ZipFile(dataset_path / "dataset.zip", "r") as zip_ref:
        zip_ref.extractall(str(dataset_path))

    # Create the dataset metadata file
    logging.info("Creating dataset metadata for %s", dataset_name)
    with open(
        _dataset_metadata_path(dataset_name), "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(
            {
                "name": dataset_name,
                "path": str(dataset_path),
                "version": "0.1.0",
            },
            metadata_file,
            ensure_ascii=False,
            indent=4,
        )


def _load_geolife_dataset(force_redownload: bool) -> Dataset:
    dataset_name = "geolife"
    # Check if the dataset needs to be downloaded
    if force_redownload or not _dataset_metadata_path(dataset_name).exists():
        _download_dataset(__GEOLIFE_DATASET_URL, dataset_name)

    # parse and yupify the dataset
    raise NotImplementedError


def _load_mnist_dataset(force_redownload: bool) -> Dataset:
    dataset_name = "mnist"
    # Check if the dataset needs to be downloaded
    if force_redownload or not _dataset_metadata_path(dataset_name).exists():
        _download_dataset(__MNIST_DATASET_URL, dataset_name)

    # parse and yupify the dataset
    raise NotImplementedError


def _load_langevin_dataset() -> Dataset:
    raise NotImplementedError


def _load_diff_diff_dataset() -> Dataset:
    raise NotImplementedError


def _load_langevin_vs_diff_diff_dataset() -> Dataset:
    raise NotImplementedError


def load_dataset(datase_name: str, force_redownload: bool = False) -> Dataset:
    """
    Loads a dataset from the datasets folder.

    Parameters
    ----------
    datase_name : str
        The name of the dataset.

        Must be one of the following:
            - geolife
            - mnist
            - langevin
            - diff_diff
            - langevin_vs_diff_diff
    force_redownload : bool, optional
        If True, the dataset will be downloaded again, even if it already exists.

        Only applies to the geolife and mnist datasets.
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
        f"The dataset {datase_name} is not supported. "
        "Supported datasets are: "
        "geolife, mnist, langevin, diff_diff, langevin_vs_diff_diff"
    )
