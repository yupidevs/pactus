from __future__ import annotations

import json
import logging
from abc import ABCMeta, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

from sklearn.model_selection import train_test_split
from yupi import Trajectory
from yupi.core import JSONSerializer

from pactus import config
from pactus.dataset._utils import _get_path, _get_progress_log


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


class Dataset(Data, metaclass=ABCMeta):
    """Wraps the data with some general properties that describes a full dataset"""

    def __init__(
        self, name: str, version: str, refetch: bool = False, reyupify: bool = False
    ):
        self.name = name
        self.version = version
        self.refetch = refetch
        self.reyupify = reyupify
        self._metadata = self._load_metadata()
        self._yupi_data: dict
        self.dir = _get_path(config.DS_DIR, self.name)
        self.raw_dir = _get_path(config.DS_RAW_DIR, self.name)

        trajs, labels = self.load()
        super().__init__(self, trajs, labels)

    def fetch(self) -> None:
        """Downloads the dataset in case needed"""

    @abstractmethod
    def yupify(self) -> Tuple[List[Trajectory], List[Any]]:
        """Parses or generates the dataset and convert it to yupi trajectories"""

    def __len__(self):
        return len(self.trajs)

    @property
    def metadata(self) -> dict:
        """Dataset metadata"""
        assert (
            self._metadata is not None
        ), f"There is not metadata loaded for the {self.name} dataset."
        return self._metadata

    @property
    def _has_metadata(self) -> bool:
        return self._metadata is not None

    def _default_metadata(self) -> dict:
        metadata = {
            "name": self.name,
            "path": str(_get_path(config.DS_METADATA_FILE, self.name)),
            "version": self.version,
            "yupi_data": None,
        }
        return metadata

    def _load_metadata(self) -> Union[dict, None]:
        metadata_path = _get_path(config.DS_METADATA_FILE, self.name)
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _refetch_required(self) -> bool:
        """Checks if the current cache requires a refetch."""
        needs_refetch = self.refetch or not self._has_metadata

        if self._has_metadata:
            stored_major = self.metadata["version"].split(".")[0]
            current_major = self.version.split(".")[0]
            needs_refetch = needs_refetch or stored_major != current_major

        return needs_refetch

    def _yupify_required(self) -> bool:
        """Checks if the current cache requires a reyupify."""
        needs_yupify = self.reyupify or not self._has_metadata

        if self._has_metadata:
            stored_minor = self.metadata["version"].split(".")[1]
            current_minor = self.version.split(".")[1]
            yupi_data_file = self.dir / "yupi_data.json"
            needs_yupify = (
                needs_yupify
                or stored_minor != current_minor
                or not yupi_data_file.exists()
            )

        if needs_yupify:
            logging.info("Yupify is required for the %s dataset", self.name)
        return needs_yupify

    def _save_json(self, path: Path, data: dict):
        with open(path, "w", encoding="utf-8") as md_file:
            json.dump(data, md_file, ensure_ascii=False, indent=4)

    def _update_metadata(self):
        logging.info("Updating metadata for %s dataset", self.name)
        metadata_path = Path(self.metadata["path"])
        self._save_json(metadata_path, self.metadata)

    def _create_folder(self):
        """Create dataset folder if not exists"""
        dataset_path = _get_path(config.DS_DIR, self.name)
        dataset_path.mkdir(parents=True, exist_ok=True)

    def _yupify(self):
        """
        Gets the trajectoires in a yupi format and the labels
        and stores them
        """
        logging.info("Yupifying %s dataset", self.name)
        trajs, labels = self.yupify()

        ds_dir = _get_path(config.DS_DIR, self.name)

        for i, traj in enumerate(trajs):
            traj.traj_id = str(i)

        json_trajs = [JSONSerializer.to_json(traj) for traj in trajs]
        yupi_data = {"trajs": json_trajs, "labels": labels}

        logging.info("Saving yupify trajectories for %s dataset", self.name)
        data_path = ds_dir / "yupi_data.json"
        self._save_json(data_path, yupi_data)
        self.metadata["yupi_data"] = str(data_path)

    def _ensure_cache(self):
        if self._refetch_required():
            logging.info("Fetching %s dataset", self.name)
            self.fetch()
            self._metadata = self._default_metadata()
            self._update_metadata()

        if self._yupify_required():
            self._yupify()

        self.metadata["version"] = self.version
        self._update_metadata()

    def _load_yupi_data(self):
        assert self.metadata["yupi_data"] is not None
        logging.info("Loading yupify data for %s dataset", self.name)
        yupi_metadata_path = self.metadata["yupi_data"]
        with open(yupi_metadata_path, "r", encoding="utf-8") as md_file:
            self._yupi_data = json.load(md_file)

    def _load(self) -> Tuple[List[Trajectory], List[Any]]:
        self._load_yupi_data()
        logging.info("Loading %s dataset", self.name)

        def _load_traj(traj, i, total):
            print(_get_progress_log(i, total), end="\r")
            return JSONSerializer.from_json(traj)

        total = len(self._yupi_data["trajs"])
        trajs = [
            _load_traj(traj, i + 1, total)
            for i, traj in enumerate(self._yupi_data["trajs"])
        ]
        labels = self._yupi_data["labels"]
        return trajs, labels

    def load(self) -> Tuple[List[Trajectory], List[Any]]:
        """Loads the dataset"""

        self._create_folder()
        self._ensure_cache()
        return self._load()
