
MNIST_DS_URL = "http://yann.lecun.com/exdb/mnist/"

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
            "yupify_metadata": None
        }
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)

