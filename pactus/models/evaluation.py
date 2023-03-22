from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from pactus import config
from pactus.dataset import Data
from pactus.dataset._utils import _get_path

MAIN_SEP = "="
SUB_SEP = "-"

LATEX_CM_ROW_TEMPLATE = Template(
    r"""            & $cls_name & $cls_vals & $cls_prec \\
"""
)

LATEX_CM_TEMPLATE = Template(
    r"""
\begin{figure}[ht]
\caption{$caption}
\vspace{2mm}
\centering
    \begin{tabular}{cc${c_cols}c}
        \toprule
        \multicolumn{2}{c}{\multirow{2}[4]{*}{\bf $model_name}} &
        \multicolumn{$cls_count}{c}{\bf Actual} &
        \multirow{2}[4]{*}{\bf Precision} \\
        & \cline{2-$c_line_top}
        & & $cls_head & \\
        \midrule
        \multirow{$cls_count}{*}{\bf Predicted}
$cls_rows            \midrule
        \multicolumn{2}{c}{\bf Recall} & $recalls \\
        \bottomrule
    \end{tabular}
\end{figure}
"""
)


class Evaluation:
    def __init__(
        self,
        dataset_name: str,
        trajs_ids: List[str],
        y_true,
        y_pred,
        model_summary: dict,
    ):
        self.dataset_name = dataset_name
        self.traj_ids = trajs_ids
        self.y_true = y_true if not isinstance(y_true, np.ndarray) else y_true.tolist()
        self.y_pred = y_pred if not isinstance(y_pred, np.ndarray) else y_pred.tolist()
        self.model_summary = model_summary
        self.classes = list(set(self.y_true))
        self.classes.sort()

        self._confusion_matrix = confusion_matrix(
            self.y_true, self.y_pred, labels=self.classes
        ).T
        pre, rec, f_sc, sup = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.classes
        )
        self.precision = np.asarray(pre)
        self.recall = np.asarray(rec)
        self.f_score = np.asarray(f_sc)
        self.support = np.asarray(sup)

        self.acc_overall = accuracy_score(self.y_true, self.y_pred, normalize=True)
        self.f1_score = f1_score(
            self.y_true, self.y_pred, average="macro", zero_division=0
        )

    @staticmethod
    def from_data(
        data: Data,
        predictions: List[Any],
        model_summary: dict,
    ) -> Evaluation:
        return Evaluation(
            dataset_name=data.dataset.name,
            trajs_ids=[traj.traj_id for traj in data.trajs if traj.traj_id is not None],
            y_true=data.labels,
            y_pred=predictions,
            model_summary=model_summary,
        )

    def _conf_matrix_perc(self) -> np.ndarray:
        c_matrix = self._confusion_matrix.astype("float")
        for i in range(c_matrix.shape[0]):
            c_matrix[:, i] /= c_matrix[:, i].sum()
        return c_matrix

    def _show_confusion_matrix(self):
        """Show the confusion matrix."""
        print("\nConfusion matrix:\n")

        # Normalize the confusion matrix by columns
        c_matrix = self._conf_matrix_perc()

        classes = list(self.classes) + ["precision"]
        col_width = max(5, max(map(len, map(str, classes[:-1])))) + 2

        print(*[f"{c:<{col_width}}".format() for c in classes], sep="")
        print(MAIN_SEP * col_width * (len(classes)))
        for i, row in enumerate(c_matrix):
            row = np.append(row, self.precision[i])
            print(*[f"{round(c * 100, 2):<{col_width}}" for c in row], sep="")
        print(SUB_SEP * col_width * (len(classes)))
        print(
            *[f"{round(rc * 100, 2):<{col_width}}" for rc in self.recall],
            sep="",
        )

    def show(self):
        """Show the evaluation results."""
        self._show_general_stats()
        self._show_confusion_matrix()

    def _show_general_stats(self):
        """Show the general statistics."""
        print("\nGeneral statistics:\n")
        print(f"Accuracy: {self.acc_overall:.3f}")
        print(f"F1-score: {self.f1_score:.3f}")
        print(f"Mean precision: {self.precision.mean():.3f}")
        print(f"Mean recall: {self.recall.mean():.3f}")

    @staticmethod
    def load(file_name: str) -> Evaluation:
        """Loads an evaluation from a file.

        Parameters
        ----------
        file_name : str
            The name of the file to load the evaluation from. It
            must end with '.json'.

        Returns
        -------
        Evaluation
            The loaded evaluation.
        """

        if not file_name.endswith(".json"):
            raise ValueError("file_name extension must be '.json'")

        with open(file_name, "r", encoding="utf-8") as data_fd:
            data = json.load(data_fd)

        assert len(data["indices"]) == len(data["y_pred"])
        ds_name = data["dataset_name"]
        indices = data["indices"]
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        summary = data["model_summary"]
        return Evaluation(
            dataset_name=ds_name,
            trajs_ids=indices,
            y_pred=y_pred,
            y_true=y_true,
            model_summary=summary,
        )

    def save(self, file_name: str):
        """Save the evaluation to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the evaluation to. It
            must end with '.json'.
        """

        if not file_name.endswith(".json"):
            raise ValueError("file_name extension must be '.json'")

        data = {
            "dataset_name": self.dataset_name,
            "indices": self.traj_ids,
            "y_pred": self.y_pred,
            "y_true": self.y_true,
            "classes": self.classes,
            "model_summary": self.model_summary,
        }

        assert len(data["indices"]) == len(data["y_pred"])

        if "/" in file_name:
            dir_name = "/".join(file_name.split("/")[:-1])
            dir_path = Path(dir_name)
            dir_path.mkdir(parents=True, exist_ok=True)

        with open(file_name, "w", encoding="utf-8") as data_fd:
            json.dump(data, data_fd, ensure_ascii=False, indent=4)

    def to_markdown(self) -> str:
        """Evaluation summary in markdown style."""
        summary = self.model_summary.copy()
        model_name = summary.pop("name")
        ans = "# Evaluation results\n\n"
        ans += f"**Dataset:** {self.dataset_name} \\\n"
        ans += f"**Model:** {model_name}\n"
        ans += "\n## Model Summary\n\n"
        for param, val in summary.items():
            ans += f"- `{param} = {val}`\n"
        ans += "\n## Confusion Matrix\n\n"

        c_matrix = self._conf_matrix_perc()

        head = (
            "| Predicted \\ Actual | "
            + " | ".join(map(str, self.classes))
            + " | Precision |\n"
        )
        sep = "| :--: | " + " | ".join([":--:" for _ in self.classes]) + " | :--: |\n"
        body = ""
        for i, row in enumerate(c_matrix):
            row = np.append(row, self.precision[i])
            str_row = [str(round(c * 100, 2)) for c in row]
            str_row[i] = f"**{str_row[i]}**"
            body += f"| **{self.classes[i]}** | " + " | ".join(str_row) + " |\n"
        recall = (
            "| **Recall** | "
            + " | ".join([str(round(rc * 100, 2)) for rc in self.recall])
            + " |\n"
        )
        ans += head + sep + body + recall
        return ans

    def to_latex(self) -> str:
        """Evaluation summary in latex style."""
        summary = self.model_summary.copy()
        model_name = summary.pop("name")
        model_name = " ".join([val.title() for val in model_name.split("_")])
        ans = cls_rows = ""
        c_matrix = self._conf_matrix_perc()
        classes = [cls.replace("_", "\\_") for cls in self.classes]

        for i, row in enumerate(c_matrix):
            str_row = [f"{str(round(c * 100, 2))} \\%" for c in row]
            str_row[i] = r"\textbf{" + str_row[i] + "}"
            row = LATEX_CM_ROW_TEMPLATE.substitute(
                cls_name=classes[i],
                cls_vals=" & ".join(str_row),
                cls_prec=f"{str(round(self.precision[i] * 100, 2))} \\%",
            )
            cls_rows += row

        ans += LATEX_CM_TEMPLATE.substitute(
            caption=f"Confusion matrix for {model_name}. Dataset: {self.dataset_name}",
            model_name=model_name,
            c_cols="c" * len(classes),
            c_line_top=str(len(classes) + 1),
            cls_count=len(classes),
            cls_head=" & ".join(classes),
            cls_rows=cls_rows,
            recalls=" & ".join(
                [f"{str(round(rc * 100, 2))} \\%" for rc in self.recall]
            ),
        )
        return ans
