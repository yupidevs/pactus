from yuca.dataset import DatasetSlice, Dataset
from yuca.dataset._utils import _get_path

import config
import json
from typing import Any
from pathlib import Path

class Evaluation:

    def __init__(self, model_summary: dict, data: Dataset | DatasetSlice, predictions: list[Any]):
        self.dataset = data.dataset
        self.trajs = data.trajs
        self.predictions = predictions
        self.model_summary = model_summary
    
    def show(self):
        raise NotImplementedError

    def save(self, file_name: str) -> Path:
        if not file_name.endswith(".json"):
            raise ValueError("file_name extension must be '.json'")

        data = {
            "indices": [int(traj.traj_id) for traj in self.trajs],
            "predictions": self.predictions,
            "model_summary": self.model_summary
        }

        file_path = _get_path(config.DS_EVALS_DIR, self.dataset.name) / file_name

        with open(file_path, "w", encoding="utf-8") as data_fd:
            json.dump(data, data_fd, ensure_ascii=False, indent=4)
        
        return file_path