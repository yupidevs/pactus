import logging
import zipfile

import requests

import yuca.config as cfg
from yuca.dataset._utils import _get_path


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
