from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import requests
from git.cmd import Git
from sklearn.model_selection import train_test_split
from yupi import Trajectory
from yupi.core import JSONSerializer
from yupi.core.featurizers import Featurizer

from pactus import config
from pactus.dataset._utils import _get_path, download_dataset

REPO_URL = "https://github.com/yupidevs/trajectory-datasets"


class Data:
    """
    Structure that groups the trajectories and labels along with some
    useful methods to work with the set of them.

    Parameters
    ----------
    trajs: List[Trajectory]
        A list that contains a subset of the dataset trajectories.
    labels: List[Any]
        A list that contains the label of each trajectory from the subset.
    dataset_name: str
        Name of the dataset where the trajectories come from. If not provided,
        it will be set to "custom".
    """

    def __init__(
        self,
        trajs: List[Trajectory],
        labels: List[Any],
        dataset_name: str = "custom",
    ) -> None:
        self.trajs = trajs
        self.labels = labels
        self.label_counts = Counter(labels)
        self.feats = None
        self.last_featurizer = None
        self.dataset_name = dataset_name

    @property
    def classes(self) -> List[Any]:
        """Classes present in the dataset."""
        return list(self.label_counts.keys())

    def __len__(self) -> int:
        return len(self.trajs)

    def featurize(self, featurizer: Featurizer) -> np.ndarray:
        """
        Featurizes the trajectories.

        Parameters
        ----------
        featurizer : Featurizer
            Featurizer to be used.

        Returns
        -------
        np.ndarray
            A numpy array with the featurized trajectories.
        """
        if self.feats is None or self.last_featurizer != featurizer:
            self.last_featurizer = featurizer
            self.feats = featurizer.featurize(self.trajs)
        assert self.feats is not None
        return self.feats

    def take(
        self,
        size: Union[float, int],
        stratify: bool = True,
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ) -> Data:
        """
        Takes a subset of the dataset.

        Parameters
        ----------
        size : Union[float, int]
            If float, it should be between 0 and 1 and it will be interpreted
            as the proportion of the dataset to be taken. If int, it should be
            between 0 and the dataset size and it will be interpreted as the
            number of trajectories to be taken.
        stratify : bool, optional
            If True, the dataset will be stratified by the labels, by default
            True.
        shuffle : bool, optional
            If True, the dataset will be shuffled before taking the subset,
            by default True.
        random_state : Union[int, None], optional
            Random state to be used, by default None.

        Returns
        -------
        Data
            A new Data object with the subset of the dataset.
        """
        if isinstance(size, int):
            assert 0 < size < len(self), "size should be within 0 and len(self)"
            size /= len(self)

        ans, _ = self.split(
            size, stratify=stratify, shuffle=shuffle, random_state=random_state
        )
        return ans

    def cut(self, size: Union[float, int]) -> Tuple[Data, Data]:
        """
        Similar to split, but without shuffle, stratify, etc. Just slices the
        dataset into two parts.

        Parameters
        ----------
        size : Union[float, int]
            If float, it should be between 0 and 1 and it will be interpreted
            as the proportion of the dataset to be taken. If int, it should be
            between 0 and the dataset size and it will be interpreted as the
            number of trajectories to be taken.

        Returns
        -------
        Tuple[Data, Data]
            A tuple with two Data objects, the first one with the first part
            of the cut and the second one with the second part.
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
        left_d = Data(left, left_labels)
        right_d = Data(right, right_labels)
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

        train_data = Data(x_train, y_train)
        test_data = Data(x_test, y_test)
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
        return Data(trajs, labels)

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
        return Data(trajs, labels)


class Dataset(Data):
    """
    Wraps the data with some general properties that describes a full dataset

    Parameters
    ----------
    name : str
        Name of the dataset.
    trajs: List[Trajectory]
        A list that contains the dataset trajectories.
    labels: List[Any]
        A list that contains the label of each trajectory.
    version: int
        Dataset version.
    """

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
        super().__init__(trajs, labels, dataset_name=name)

    def __len__(self):
        return len(self.trajs)

    @staticmethod
    def _from_json(name: str, data: dict) -> Dataset:
        assert "trajs" in data, "trajs not found in dataset"
        assert "labels" in data, "labels not found in dataset"
        trajs = [JSONSerializer.from_json(traj) for traj in data["trajs"]]
        return Dataset(
            name=name,
            trajs=trajs,
            labels=data["labels"],
            version=data.get("version", 0),
        )

    @staticmethod
    def _get_dataset_url(name: str) -> str:
        g_cmd = Git()
        output = g_cmd.ls_remote(REPO_URL, sort="-v:refname", tags=True)
        tags = output.split("\n")[0].split("/")[-1]
        tags = [ref.split("/")[-1] for ref in output.split("\n")]
        for tag in tags:
            url = f"{REPO_URL}/releases/download/{tag}/{name}.zip"
            if requests.head(url).status_code == 302:
                return url
        assert False, "Could not find the given dataset"

    @staticmethod
    def _from_url(name: str, force: bool = False) -> Dataset:
        url = Dataset._get_dataset_url(name)
        raw_dir = _get_path(config.DS_DIR, name)
        yupi_data_file = raw_dir / f"{name}.json"

        if not force and yupi_data_file.exists():
            with open(yupi_data_file, "r", encoding="utf-8") as yupi_fd:
                data = json.load(yupi_fd)
                if "trajs" in data and "labels" in data:
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
        return Dataset.get("geolife", redownload=redownload)

    @staticmethod
    def animals(redownload: bool = False) -> Dataset:
        """Loads the animals dataset"""
        return Dataset.get("animals", redownload=redownload)

    @staticmethod
    def mnist_stroke(redownload: bool = False) -> Dataset:
        """Loads the mnist_stroke dataset"""
        return Dataset.get("mnist_stroke", redownload=redownload)

    @staticmethod
    def hurdat2(redownload: bool = False) -> Dataset:
        """Loads the hurdat2 dataset"""
        return Dataset.get("hurdat2", redownload=redownload)

    @staticmethod
    def cma_bst(redownload: bool = False) -> Dataset:
        """Loads the cma_bst dataset"""
        return Dataset.get("cma_bst", redownload=redownload)

    @staticmethod
    def uci_gotrack(redownload: bool = False) -> Dataset:
        """Loads the uci_gotrack dataset"""
        return Dataset.get("uci_gotrack", redownload=redownload)

    @staticmethod
    def uci_movement_libras(redownload: bool = False) -> Dataset:
        """Loads the uci_movement_libras dataset"""
        return Dataset.get("uci_movement_libras", redownload=redownload)

    @staticmethod
    def uci_pen_digits(redownload: bool = False) -> Dataset:
        """Loads the uci_pen_digits dataset"""
        return Dataset.get("uci_pen_digits", redownload=redownload)

    @staticmethod
    def uci_characters(redownload: bool = False) -> Dataset:
        """Loads the uci_characters dataset"""
        return Dataset.get("uci_characters", redownload=redownload)

    @staticmethod
    def traffic(redownload: bool = False) -> Dataset:
        """Loads the traffic dataset"""
        return Dataset.get("traffic", redownload=redownload)

    @staticmethod
    def diffusive_particles(redownload: bool = False) -> Dataset:
        """Loads the diffusive particles dataset"""
        return Dataset.get("diffusive_particles", redownload=redownload)

    @staticmethod
    def get(dataset_name: str, redownload: bool = False) -> Dataset:
        """Loads a dataset from the trajectory-dataset repository"""
        return Dataset._from_url(dataset_name, force=redownload)
