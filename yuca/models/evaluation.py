import json
from pathlib import Path
from typing import Any

from yuca import config
from yuca.dataset import Dataset, DatasetSlice
from yuca.dataset._utils import _get_path


class Evaluation:
    def __init__(
        self, model_summary: dict, data: Dataset | DatasetSlice, predictions: list[Any]
    ):
        self.dataset = data.dataset
        self.trajs = data.trajs
        self.predictions = predictions
        self.model_summary = model_summary

    def show(self):
        """Show the evaluation results."""
        raise NotImplementedError

    def save(self, file_name: str) -> Path:
        """Save the evaluation to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the evaluation to. It
            must end with '.json'.

        Returns
        -------
        Path
            The path to the saved file.
        """

        if not file_name.endswith(".json"):
            raise ValueError("file_name extension must be '.json'")

        data = {
            "indices": [
                int(traj.traj_id) for traj in self.trajs if traj.traj_id is not None
            ],
            "predictions": self.predictions,
            "model_summary": self.model_summary,
        }

        assert len(data["indices"]) == len(data["predictions"])

        file_path = _get_path(config.DS_EVALS_DIR, self.dataset.name) / file_name

        with open(file_path, "w", encoding="utf-8") as data_fd:
            json.dump(data, data_fd, ensure_ascii=False, indent=4)

        return file_path
