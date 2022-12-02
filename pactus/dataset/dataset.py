from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

from git.cmd import Git
from sklearn.model_selection import train_test_split
from yupi import Trajectory
from yupi.core import JSONSerializer

from pactus import config
from pactus.dataset._utils import _get_path, download_dataset

REPO_URL = "https://github.com/yupidevs/trajectory-datasets"


class Data:
    """
    Structure that groups the trajectories and labels along with some
    useful methods to work with the set of them.
    """

    def __init__(
        self, dataset: Dataset, trajs: List[Trajectory], labels: List[Any]
    ) -> None:
        self.dataset = dataset
        self.trajs = trajs
        self.labels = labels
        self.label_counts = Counter(labels)

    @property
    def classes(self) -> List[Any]:
        """Classes present in the dataset."""
        return list(self.label_counts.keys())

    def __len__(self) -> int:
        return len(self.trajs)

    def take(
        self,
        size: Union[float, int],
        stratify: bool = True,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ) -> Data:
        """Takes a subset of the dataset."""
        if isinstance(size, int):
            assert 0 < size < len(self), "size should be within 0 and len(self)"
            size /= len(self)

        ans, _ = self.split(
            size, stratify=stratify, shuffle=shuffle, random_state=random_state
        )
        return ans

    def cut(self, size: Union[float, int]):
        """
        Similar to split, but without shuffle, stratify, etc. Just slices the
        dataset into two parts.
        """
        if isinstance(size, float):
            assert 0 < size < 1, "size should be within 0 and 1 if float"
            size = int(len(self) * size)
        else:
            assert (
                0 < size < len(self)
            ), "size should be within 0 and the dataset size if int"

        left, right = self.trajs[:size], self.trajs[size:]
        left_labels, right_labels = self.labels[:size], self.labels[size:]
        left_d = Data(self.dataset, left, left_labels)
        right_d = Data(self.dataset, right, right_labels)
        return left_d, right_d

    def split(
        self,
        train_size: Union[float, int] = 0.8,
        stratify: bool = True,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ) -> Tuple[Data, Data]:
        """
        Splits the dataset into train and test dataset slices.

        It uses the sklearn.model_selection.train_test_split function.

        Parameters
        ----------
        train_size : Union[float, int], optional
            The proportion of the dataset to include in the train split.
            If float, should be between 0.0 and 1.0, if int, represents the
            absolute number of train samples. By default 0.8.
        stratify : bool, optional
            If True, the split will be stratified according to the labels,
            by default True
        shuffle : bool, optional
            If True, the split will be shuffled, by default True
        random_state : Union[int, None], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Tuple[Data, Data]
            A tuple with the train and test Data objects.
        """
        if isinstance(train_size, int):
            assert (
                0 < train_size < len(self)
            ), "train_size should be within 0 and the dataset size if int"
            train_size /= len(self)
        else:
            assert 0 < train_size < 1, "train_size should be within 0 and 1 if float"

        x_train, x_test, y_train, y_test = train_test_split(
            self.trajs,
            self.labels,
            train_size=train_size,
            stratify=self.labels if stratify else None,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_data = Data(self.dataset, x_train, y_train)
        test_data = Data(self.dataset, x_test, y_test)
        return train_data, test_data

    def map(self, func: Callable[[Trajectory, Any], Tuple[Trajectory, Any]]) -> Data:
        """
        Applies a function to each trajectory and label pair.

        Usefull to apply some preprocessing to the trajectories
        or the labels.

        Parameters
        ----------
        func : Callable[[Trajectory, Any], Tuple[Trajectory, Any]]
            Function to be applied to each trajectory and label pair.

        Returns
        -------
        Data
            A new Data object with the results of the function.
        """
        trajs, labels = [], []
        for traj, label in zip(self.trajs, self.labels):
            traj, label = func(traj, label)
            trajs.append(traj)
            labels.append(label)
        return Data(self.dataset, trajs, labels)

    def filter(self, func: Callable[[Trajectory, Any], bool]) -> Data:
        """
        Filters the dataset based on a function.

        Parameters
        ----------
        func : Callable[[Trajectory, Any], bool]
            Function to be applied to each trajectory and label pair.

        Returns
        -------
        Data
            A new Data object with the filtered trajectories and labels.
        """
        trajs, labels = [], []
        for traj, label in zip(self.trajs, self.labels):
            if func(traj, label):
                trajs.append(traj)
                labels.append(label)
        logging.info("Filtered %d of %d trajectories", len(trajs), len(self))
        return Data(self.dataset, trajs, labels)


class Dataset(Data):
    """Wraps the data with some general properties that describes a full dataset"""

    _last_tag = None

    def __init__(
        self,
        name: str,
        trajs: List[Trajectory],
        labels: List[Any],
        version: int = 0,
    ):
        self.name = name
        self.version = version
        self.trajs = trajs
        self.labels = labels
        super().__init__(self, trajs, labels)

    def __len__(self):
        return len(self.trajs)

    @staticmethod
    def _from_json(name: str, data: dict) -> Dataset:
        assert "trajs" in data, "trajs not found in dataset"
        assert "labels" in data, "labels not found in dataset"
        assert "version" in data, "version not found in dataset"
        trajs = [JSONSerializer.from_json(traj) for traj in data["trajs"]]
        return Dataset(
            name=name,
            trajs=trajs,
            labels=data["labels"],
            version=data["version"],
        )

    @staticmethod
    def _get_dataset_url(name: str) -> str:
        tag = Dataset._last_tag
        if tag is None:
            g_cmd = Git()
            output = g_cmd.ls_remote(REPO_URL, sort="-v:refname", tags=True)
            tag = output.split("\n")[0].split("/")[-1]
            Dataset._last_tag = tag
        assert tag is not None, "Could not find the last tag"
        return f"{REPO_URL}/releases/download/{tag}/{name}.zip"

    @staticmethod
    def _from_url(name: str, force: bool = False) -> Dataset:
        url = Dataset._get_dataset_url(name)
        raw_dir = _get_path(config.DS_DIR, name)
        yupi_data_file = raw_dir / f"{name}.json"

        if not force and yupi_data_file.exists():
            with open(yupi_data_file, "r", encoding="utf-8") as yupi_fd:
                data = json.load(yupi_fd)
                if "trajs" in data and "labels" in data and "version" in data:
                    return Dataset._from_json(name, data)
            logging.warning("Invalid dataset file, downloading again")

        download_dataset(url, name)
        return Dataset.from_file(yupi_data_file, name)

    @staticmethod
    def from_file(path: Union[Path, str], name: str) -> Dataset:
        """Loads a dataset from a file."""
        _path = path if isinstance(path, Path) else Path(path)
        with open(_path, "r", encoding="utf-8") as yupi_fd:
            data = json.load(yupi_fd)
            return Dataset._from_json(name, data)

    @staticmethod
    def geolife(redownload: bool = False) -> Dataset:
        """Loads the geolife dataset"""
        return Dataset._from_url("geolife", force=redownload)

    @staticmethod
    def animals(redownload: bool = False) -> Dataset:
        """Loads the animals dataset"""
        return Dataset._from_url("animals", force=redownload)

    @staticmethod
    def mnist_stroke(redownload: bool = False) -> Dataset:
        """Loads the mnist_stroke dataset"""
        return Dataset._from_url("mnist_stroke", force=redownload)

    @staticmethod
    def hurdat2(redownload: bool = False) -> Dataset:
        """Loads the hurdat2 dataset"""
        return Dataset._from_url("hurdat2", force=redownload)

    @staticmethod
    def cma_bst(redownload: bool = False) -> Dataset:
        """Loads the cma_bst dataset"""
        return Dataset._from_url("cma_bst", force=redownload)

    @staticmethod
    def uci_gotrack(redownload: bool = False) -> Dataset:
        """Loads the uci_gotrack dataset"""
        return Dataset._from_url("uci_gotrack", force=redownload)

    @staticmethod
    def uci_movement_libras(redownload: bool = False) -> Dataset:
        """Loads the uci_movement_libras dataset"""
        return Dataset._from_url("uci_movement_libras", force=redownload)

    @staticmethod
    def uci_pen_digits(redownload: bool = False) -> Dataset:
        """Loads the uci_pen_digits dataset"""
        return Dataset._from_url("uci_pen_digits", force=redownload)

    @staticmethod
    def uci_characters(redownload: bool = False) -> Dataset:
        """Loads the uci_characters dataset"""
        return Dataset._from_url("uci_characters", force=redownload)

    @staticmethod
    def stochastic_models(redownload: bool = False) -> Dataset:
        """Loads the stochastic models dataset"""
        return Dataset._from_url("stochastic_models", force=redownload)
