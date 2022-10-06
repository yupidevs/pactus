import json

from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod
from random import shuffle
from sklearn.model_selection import train_test_split

from yupi import Trajectory, JSONSerializer
from yuca import config
from yuca.datasets._utils import _get_path

class Dataset(ABC):
    """Class for a dataset."""

    def __init__(self, name: str, version: str, refetch: bool = False, reyupify: bool = False):
        self.name = name
        self.version = version
        self.refetch = refetch
        self.reyupify = reyupify
        self.path = _get_path(config.DS_DIR, self.name)
        self.metadata = self._load_metadata()
        self.yupi_metadata: dict

        self.labels, self.trajs = self.load()
        self.classes = set(self.labels)

    def _fetch(self, dataset_folder: Path) -> None:
        """Downloads the dataset in case needed"""
        pass
    
    @abstractmethod
    def _yupify(self) -> tuple[list[Trajectory], list[Any]]:
        """Parses the dataset and convert it to yupi trajectories"""

    def __len__(self):
        return len(self.trajs)

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
            needs_yupify = needs_yupify or stored_minor != current_minor or self.metadata["yupify_metadata"] is None

        return needs_yupify

    def _update_metadata(self):
        metadata_path = _get_path(config.DS_METADATA_FILE, self.name)
        with open(metadata_path, "w", encoding="utf-8") as md_file:
            json.dump(self.metadata, md_file, ensure_ascii=False, indent=4)

    def _create_folder(self):
        """ Create dataset folder if not exists """
        dataset_path = _get_path(config.DS_DIR, self.name)
        dataset_path.mkdir(parents=True, exist_ok=True)

    def yupify(self):
        trajs, labels = self._yupify()

        trajs_paths = []
        yupi_dir = _get_path(config.DS_YUPI_DIR, self.name)
        for i, traj in enumerate(trajs):
            traj_path = str(yupi_dir / f"traj_{i}.json")
            JSONSerializer.save(traj, traj_path, overwrite=True)
            trajs_paths.append(traj_path)

        yupify_metadata = {
            "trajs_paths": trajs_paths,
            "labels": labels
        }

        metadata_path = str(yupi_dir / "yupify_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as md_file:
            json.dump(yupify_metadata, md_file, ensure_ascii=False, indent=4)

        self.metadata["yupify_metadata"] = metadata_path

    def _ensure_cache(self):
        if self._refetch_required():
            self._fetch(_get_path(config.DS_RAW_DIR, self.name))
        
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
        trajs = [JSONSerializer.load(traj) for traj in self.yupi_metadata["trajs_paths"]]
        labels = self.yupi_metadata["labels"]
        return trajs, labels
    
    def load(self) -> tuple[list[Trajectory], list[Any]]:
        """Loads the dataset"""
        
        self._create_folder()
        self._ensure_cache()
        return self._load()
    
    def split(self, train_size: float) -> tuple[Dataset, Dataset]:
        assert 0 < train_size < 1, "train_size should be within 0 and 1"

        x_train, y_train, x_test, y_test = train_test_split