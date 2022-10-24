import logging
from pathlib import Path
from string import Template
from typing import List, Tuple

import numpy as np
from yupi import Trajectory

from pactus.dataset import Dataset, download_dataset

# Dataset metadata
NAME = "mnist_stroke"
VERSION = "0.1.0"  # See version description in config.py
DOWNLOAD_URL = (
    "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/"
    "raw/master/sequences.tar.gz"
)
TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


class MnistStrokeDataset(Dataset):
    """Class for the MNIST stroke sequence dataset."""

    def __init__(self, redownload: bool = False, reyupify: bool = False):
        super().__init__(NAME, VERSION, redownload, reyupify)

    def fetch(self) -> None:
        download_dataset(DOWNLOAD_URL, self.name)
        download_dataset(TRAIN_LABELS_URL, self.name)
        download_dataset(TEST_LABELS_URL, self.name)

    def _read_labels(self, label_file: Path, count: int) -> List[str]:
        with open(label_file, "rb") as f:
            f.read(8)  # discard header info
            return [str(l) for l in f.read(count)]

    def _read_traj(self, trajectory_file: Path) -> Trajectory:
        # Load raw data from the stroke
        traj = np.loadtxt(trajectory_file, delimiter=",", skiprows=1)

        # Filter tokens from changes
        points = [(i, j) for i, j in traj if i >= 0 and j >= 0]

        return Trajectory(points=points)

    def _read_trajs(
        self, sequence_folder: Path, template: Template, count: int
    ) -> List[Trajectory]:
        trajs = []
        for i in range(count):
            traj_path = sequence_folder / template.substitute(id=i)
            trajs.append(self._read_traj(traj_path))

        return trajs

    def _yupify_mnist(
        self, sequence_folder: Path, label_file: Path, template: Template, count: int
    ) -> Tuple[List[Trajectory], List[str]]:
        """Yupifies a part of the dataset"""

        labels = self._read_labels(label_file, count)
        trajs = self._read_trajs(sequence_folder, template, count)

        return trajs, labels

    def yupify(self) -> Tuple[List[Trajectory], List[str]]:
        # Loads the raw data and preprocess it
        logging.info("Preprocessing MNIST stroke raw data")
        sequence_path = self.raw_dir / "sequences"
        train_labels_path = self.raw_dir / "train-labels-idx1-ubyte"
        test_labels_path = self.raw_dir / "t10k-labels-idx1-ubyte"

        logging.info("Yupifying train dataset...")
        train_template = Template("trainimg-$id-points.txt")
        trajs_train, labels_train = self._yupify_mnist(
            sequence_path, train_labels_path, template=train_template, count=60000
        )

        logging.info("Yupifying test dataset...")
        test_template = Template("testimg-$id-points.txt")
        trajs_test, labels_test = self._yupify_mnist(
            sequence_path, test_labels_path, template=test_template, count=10000
        )

        return trajs_train + trajs_test, labels_train + labels_test
