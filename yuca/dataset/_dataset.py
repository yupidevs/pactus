from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from yupi import Trajectory
from yupi.core import JSONSerializer

from yuca import config
from yuca.dataset._utils import _get_path, _get_progress_log


class Dataset(ABC):
    """Class for a dataset."""

    def __init__(
        self, name: str, version: str, refetch: bool = False, reyupify: bool = False
    ):
        self.dataset = self
        self.name = name
        self.version = version
        self.refetch = refetch
        self.reyupify = reyupify
        self.path = _get_path(config.DS_DIR, self.name)
        self._metadata = self._load_metadata()
        self.yupi_metadata: dict
        self.dataset_dir = _get_path(config.DS_DIR, self.name)
        self.dataset_raw_dir = _get_path(config.DS_RAW_DIR, self.name)

        self.trajs, self.labels = self.load()
        self.classes = set(self.labels)

    def fetch(self) -> None:
        """Downloads the dataset in case needed"""

    @abstractmethod
    def yupify(self) -> tuple[list[Trajectory], list[Any]]:
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
            "yupify_metadata": None,
        }
        return metadata

    def _load_metadata(self) -> dict | None:
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
            needs_yupify = (
                needs_yupify
                or stored_minor != current_minor
                or self.metadata["yupify_metadata"] is None
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

        trajs_paths = []
        yupi_dir = _get_path(config.DS_YUPI_DIR, self.name)
        ds_dir = _get_path(config.DS_DIR, self.name)

        logging.info("Saving yupified trajectories for %s dataset", self.name)
        for i, traj in enumerate(trajs):
            traj.traj_id = str(i)
            traj_path = str(yupi_dir / f"traj_{i}.json")
            JSONSerializer.save(traj, traj_path, overwrite=True)
            trajs_paths.append(traj_path)
            print(_get_progress_log(i + 1, len(trajs)), end="\r")

        yupify_metadata = {"trajs_paths": trajs_paths, "labels": labels}

        logging.info("Saving yupify metadata for %s dataset", self.name)
        metadata_path = ds_dir / "yupify_metadata.json"
        self._save_json(metadata_path, yupify_metadata)
        self.metadata["yupify_metadata"] = str(metadata_path)

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

    def _load_yupify_metadata(self):
        assert self.metadata["yupify_metadata"] is not None
        logging.info("Loading yupify metadata for %s dataset", self.name)
        yupi_metadata_path = self.metadata["yupify_metadata"]
        with open(yupi_metadata_path, "r", encoding="utf-8") as md_file:
            self.yupi_metadata = json.load(md_file)

    def _load(self) -> tuple[list[Trajectory], list[Any]]:
        self._load_yupify_metadata()
        logging.info("Loading %s dataset", self.name)

        def _load_traj(traj_path, i, total):
            print(_get_progress_log(i, total), end="\r")
            return JSONSerializer.load(traj_path)

        total = len(self.yupi_metadata["trajs_paths"])
        trajs = [
            _load_traj(traj, i + 1, total)
            for i, traj in enumerate(self.yupi_metadata["trajs_paths"])
        ]
        labels = self.yupi_metadata["labels"]
        return trajs, labels

    def load(self) -> tuple[list[Trajectory], list[Any]]:
        """Loads the dataset"""

        self._create_folder()
        self._ensure_cache()
        return self._load()

    def split(
        self,
        train_size: float,
        stratify: bool = True,
        random_state: int | None = None,
    ) -> tuple[DatasetSlice, DatasetSlice]:
        """Splits the dataset into train and test dataset slices"""
        assert 0 < train_size < 1, "train_size should be within 0 and 1"

        x_train, x_test, y_train, y_test = train_test_split(
            self.trajs,
            self.labels,
            stratify=self.labels if stratify else None,
            random_state=random_state,
        )

        train_slice = DatasetSlice(self, x_train, y_train)
        test_slice = DatasetSlice(self, x_test, y_test)
        return train_slice, test_slice


class DatasetSlice:
    """A slice of a dataset"""

    def __init__(
        self,
        dataset: Dataset,
        trajs: list[Trajectory],
        labels: list[Any],
    ):
        self.dataset = dataset
        self.trajs = trajs
        self.labels = labels

    def __len__(self):
        return len(self.trajs)
