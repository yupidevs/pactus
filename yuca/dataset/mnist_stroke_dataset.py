from yuca.dataset import Dataset
from yuca.dataset._utils import download_dataset

# Dataset metadata
NAME = "mnist_stroke"
VERSION = "0.1.0"  # See version description in config.py
DOWNLOAD_URL = (
    "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/"
    "raw/master/sequences.tar.gz"
)


class MnistStrokeDataset(Dataset):
    """Class for the MNIST stroke sequence dataset."""

    def __init__(self, redownload: bool = False, reyupify: bool = False):
        super().__init__(NAME, VERSION, redownload, reyupify)

    def fetch(self) -> None:
        download_dataset(DOWNLOAD_URL, self.name)

    def yupify(self):
        raise NotImplementedError
