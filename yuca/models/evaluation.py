import json
from pathlib import Path
from typing import Any, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from yuca import config
from yuca.dataset import Data
from yuca.dataset._utils import _get_path

MAIN_SEP = "="
SUB_SEP = "-"


class Evaluation:
    def __init__(
        self,
        model_summary: dict,
        data: Data,
        predictions: List[Any],
    ):
        self.dataset = data.dataset
        self.trajs = data.trajs
        self.y_true = data.labels
        self.y_pred = predictions
        self.model_summary = model_summary
        self.classes = list(set(self.y_true))

        self._confusion_matrix = confusion_matrix(
            self.y_true, self.y_pred, labels=self.classes
        )
        (
            self.precision,
            self.recall,
            self.f_score,
            self.support,
        ) = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.classes
        )

    def _show_confusion_matrix(self):
        """Show the confusion matrix."""
        print("\nConfusion matrix:\n")

        # Normalize the confusion matrix by columns
        self._confusion_matrix = self._confusion_matrix.astype("float")
        for i in range(self._confusion_matrix.shape[0]):
            self._confusion_matrix[:, i] /= self._confusion_matrix[:, i].sum()

        # Round the values
        # cm = np.round(cm, decimals=2)
        classes = list(self.classes) + ["precision"]
        col_width = 12

        print(*[f"{c:<12}".format() for c in classes], sep="")
        print(MAIN_SEP * col_width * (len(classes)))
        for i, row in enumerate(self._confusion_matrix):
            # get the precision
            row = np.append(row, self.precision[i])
            print(*[f"{round(c * 100, 2):<12}" for c in row], sep="")
        print(SUB_SEP * col_width * (len(classes)))
        print(
            *[f"{round(rc * 100, 2):<12}" for rc in self.recall],
            sep="",
        )

    def show(self):
        """Show the evaluation results."""
        print()
        print("Avg. Precision:", np.mean(self.precision))
        print("Avg. Recall:", np.mean(self.recall))
        self._show_confusion_matrix()

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
            "predictions": self.y_pred,
            "model_summary": self.model_summary,
        }

        assert len(data["indices"]) == len(data["predictions"])

        file_path = _get_path(config.DS_EVALS_DIR, self.dataset.name) / file_name

        with open(file_path, "w", encoding="utf-8") as data_fd:
            json.dump(data, data_fd, ensure_ascii=False, indent=4)

        return file_path
