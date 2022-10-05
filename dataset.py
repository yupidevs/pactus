from dataclasses import dataclass
from typing import Any

from yupi import Trajectory


@dataclass
class Dataset:
    """Class for a dataset."""

    name: str
    path: str
    version: str
    size: str
    classes: list[Any]
    labels: list[Any]
    trajs: list[Trajectory]

    def __len__(self):
        return len(self.trajs)
