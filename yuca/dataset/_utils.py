import logging
import zipfile
from pathlib import Path

import requests
from requests import Response

import yuca.config as cfg


def _get_path(path: str, *args) -> Path:
    return Path(path.format(*args))


def _get_progress_log(part, total):
    # If the total is unknown, just return the part
    if total == -1:
        return f"Downloaded: {part / 1024 ** 2:.2f} MB"

    passed = "#" * int(cfg.PROGRESS_BAR_LENGTH * part / total)
    rest = " " * (cfg.PROGRESS_BAR_LENGTH - len(passed))
    return f"[{passed}{rest}] {part * 100/total:.2f}%"


def _create_dataset_path(dataset_name: str) -> Path:
    logging.info("Creating dataset folder for %s", dataset_name)
    dataset_path = _get_path(cfg.DS_DIR, dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path


def _start_download(url: str, dataset_name: str) -> Response:
    logging.info("Downloading %s dataset", dataset_name)
    response = requests.get(url, allow_redirects=True, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset {dataset_name}")
    return response


def _download_until_finish(url: str, response: Response, dataset_path: Path) -> Path:
    data_length = int(response.headers.get("content-length", -1))
    dataset_file_path = dataset_path / url.split("/")[-1]
    with open(dataset_file_path, "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=cfg.DOWNLOAD_CHUNCK_SIZE):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(_get_progress_log(downloaded, data_length), end="\r")
    return dataset_file_path


def _download(url: str, dataset_name: str, dataset_path: Path) -> Path:
    # Make the download request
    response = _start_download(url, dataset_name)

    # Download the dataset to a zip file
    return _download_until_finish(url, response, dataset_path)


def _uncompress(dataset_name: str, dataset_file_path: Path, dataset_path: Path):
    logging.info("Extracting %s dataset", dataset_name)
    with zipfile.ZipFile(dataset_file_path, "r") as zip_ref:
        zip_ref.extractall(str(dataset_path))


def download_dataset(url: str, dataset_name: str) -> None:
    """Downloads a dataset from a url."""

    # Create the dataset folder if it doesn't exist
    dataset_path = _create_dataset_path(dataset_name)

    # Download the compressed version of the dataset
    dataset_file_path = _download(url, dataset_name, dataset_path)

    # Extract the dataset
    _uncompress(dataset_name, dataset_file_path, dataset_path)
