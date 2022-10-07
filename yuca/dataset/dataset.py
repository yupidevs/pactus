from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from yuca import config
from yuca.dataset._utils import _get_path
from yupi import Trajectory
from yupi.core import JSONSerializer


class Dataset(ABC):
    """Class for a dataset."""

    def __init__(
        self, name: str, version: str, refetch: bool = False, reyupify: bool = False
    ):
        self.name = name
        self.version = version
        self.refetch = refetch
        self.reyupify = reyupify
        self.path = _get_path(config.DS_DIR, self.name)
        self._metadata = self._load_metadata()
        self.yupi_metadata: dict

        self.labels, self.trajs = self.load()
        self.classes = set(self.labels)

    def fetch(self, dataset_folder: Path) -> None:
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
        needs_refetch = self.refetch or self.metadata is None

        if self.metadata is not None:
            stored_major = self.metadata["version"].split(".")[0]
            current_major = self.version.split(".")[0]
            needs_refetch = needs_refetch or stored_major != current_major

        return needs_refetch

    def _yupify_required(self) -> bool:
        """Checks if the current cache requires a reyupify."""
        needs_yupify = self.reyupify or self.metadata is None

        if self.metadata is not None:
            stored_minor = self.metadata["version"].split(".")[1]
            current_minor = self.version.split(".")[1]
            needs_yupify = (
                needs_yupify
                or stored_minor != current_minor
                or self.metadata["yupify_metadata"] is None
            )

        return needs_yupify

    def _update_metadata(self):
        metadata_path = self.metadata["path"]
        with open(metadata_path, "w", encoding="utf-8") as md_file:
            json.dump(self.metadata, md_file, ensure_ascii=False, indent=4)

    def _create_folder(self):
        """Create dataset folder if not exists"""
        dataset_path = _get_path(config.DS_DIR, self.name)
        dataset_path.mkdir(parents=True, exist_ok=True)

    def _yupify(self):
        """
        Gets the trajectoires in a yupi format and the labels
        and stores them
        """
        trajs, labels = self.yupify()

        trajs_paths = []
        yupi_dir = _get_path(config.DS_YUPI_DIR, self.name)
        for i, traj in enumerate(trajs):
            traj_path = str(yupi_dir / f"traj_{i}.json")
            JSONSerializer.save(traj, traj_path, overwrite=True)
            trajs_paths.append(traj_path)

        yupify_metadata = {"trajs_paths": trajs_paths, "labels": labels}

        metadata_path = str(yupi_dir / "yupify_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as md_file:
            json.dump(yupify_metadata, md_file, ensure_ascii=False, indent=4)

        self.metadata["yupify_metadata"] = metadata_path

    def _ensure_cache(self):
        if self._refetch_required():
            self.fetch(_get_path(config.DS_RAW_DIR, self.name))
            self._metadata = self._default_metadata()
            self._update_metadata()

        if self._yupify_required():
            self.yupify()

        self.metadata["version"] = self.version
        self._update_metadata()

    def _load_yupify_metadata(self):
        assert self.metadata["yupify_metadata"] is not None
        yupi_metadata_path = self.metadata["yupify_metadata"]
        with open(yupi_metadata_path, "w", encoding="utf-8") as md_file:
            self.yupi_metadata = json.load(md_file)

    def _load(self) -> tuple[list[Trajectory], list[Any]]:
        self._load_yupify_metadata()
        trajs = [
            JSONSerializer.load(traj) for traj in self.yupi_metadata["trajs_paths"]
        ]
        labels = self.yupi_metadata["labels"]
        return trajs, labels

    def load(self) -> tuple[list[Trajectory], list[Any]]:
        """Loads the dataset"""

        self._create_folder()
        self._ensure_cache()
        return self._load()

    def split(self, train_size: float) -> tuple[Dataset, Dataset]:
        assert 0 < train_size < 1, "train_size should be within 0 and 1"

        raise NotImplementedError
