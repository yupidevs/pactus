import gzip
import logging
import tarfile
import zipfile
from pathlib import Path

import requests
from requests import Response

import pactus.config as cfg


def _get_path(path: str, *args) -> Path:
    return Path(path.format(*args))


def _get_progress_log(part, total):
    # If the total is unknown, just return the part
    if total == -1:
        return f"Downloaded: {part / 1024 ** 2:.2f} MB"

    passed = "=" * int(cfg.PROGRESS_BAR_LENGTH * part / total)
    rest = " " * (cfg.PROGRESS_BAR_LENGTH - len(passed))
    p_bar = f"[{passed}{rest}] {part * 100/total:.2f}%"
    if part == total:
        p_bar += "\n"
    return p_bar


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
    size_mb_msg = (
        f"    Size: {data_length / 1024 ** 2:.2f} MB" if data_length != -1 else ""
    )
    dataset_file_path = dataset_path / url.split("/")[-1]
    with open(dataset_file_path, "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=cfg.DOWNLOAD_CHUNCK_SIZE):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(
                    _get_progress_log(downloaded, data_length) + size_mb_msg, end="\r"
                )
    return dataset_file_path


def _download(url: str, dataset_name: str, dataset_path: Path) -> Path:
    # Make the download request
    response = _start_download(url, dataset_name)

    # Chekc if the dataset is already downloaded
    dataset_file_path = dataset_path / url.split("/")[-1]
    if dataset_file_path.exists():
        size = -1
        if "content-length" in response.headers:
            size = int(response.headers["content-length"])
            if size == dataset_file_path.stat().st_size:
                logging.info("Dataset already downloaded")
                return dataset_file_path

        msg = (
            "There is a downloaded dataset with the same name, but the download "
            "size is unknown."
        )

        if size != -1:
            msg = (
                "It seems that the dataset is already downloaded, but the size is "
                f"different.\n"
                f"    Found: {dataset_file_path.stat().st_size / 1024 ** 2:.2f} MB\n"
                f"    Expected: {size / 1024 ** 2:.2f} MB\n"
            )
        logging.warning(msg)
        ans = input("Do you want to overwrite it? [y/n]: ")
        if ans.lower() != "y":
            return dataset_file_path

    # Download the dataset to a zip file
    return _download_until_finish(url, response, dataset_path)


def _uncompress(dataset_name: str, dataset_file_path: Path, dataset_path: Path):
    logging.info("Extracting %s dataset", dataset_name)

    str_path = str(dataset_file_path)
    if str_path.endswith(".zip"):
        with zipfile.ZipFile(dataset_file_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
    elif str_path.endswith(".tar.gz"):
        with tarfile.open(dataset_file_path, "r") as tar_ref:
            tar_ref.extractall(dataset_path)
    elif str_path.endswith(".gz"):
        with gzip.open(dataset_file_path, "rb") as gz_ref:
            with open(str(dataset_file_path.with_suffix("")), "wb") as uncompressed:
                uncompressed.write(gz_ref.read())


def download_dataset(url: str, dataset_name: str) -> None:
    """Downloads a dataset from a url."""

    if not url.endswith(".zip") and not url.endswith(".gz"):
        raise ValueError("The dataset must be a zip or .gz file")

    # Create the dataset folder if it doesn't exist
    dataset_path = _create_dataset_path(dataset_name)

    # Download the compressed version of the dataset
    dataset_file_path = _download(url, dataset_name, dataset_path)

    # Extract the dataset
    _uncompress(dataset_name, dataset_file_path, dataset_path)
