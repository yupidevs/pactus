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
        ).T
        pre, rec, f_sc, sup = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.classes
        )
        self.precision = np.asarray(pre)
        self.recall = np.asarray(rec)
        self.f_score = np.asarray(f_sc)
        self.support = np.asarray(sup)

        self.acc_overall = accuracy_score(self.y_true, self.y_pred, normalize=True)
        self.f1_score = f1_score(self.y_true, self.y_pred, average="weighted")

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
        col_width = 12

        print(*[f"{c:<12}".format() for c in classes], sep="")
        print(MAIN_SEP * col_width * (len(classes)))
        for i, row in enumerate(c_matrix):
            row = np.append(row, self.precision[i])
            print(*[f"{round(c * 100, 2):<12}" for c in row], sep="")
        print(SUB_SEP * col_width * (len(classes)))
        print(
            *[f"{round(rc * 100, 2):<12}" for rc in self.recall],
            sep="",
        )

    def show(self):
        """Show the evaluation results."""
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

    def to_markdown(self) -> str:
        """Evaluation summary in markdown style."""
        summary = self.model_summary.copy()
        model_name = summary.pop("name")
        ans = "# Evaluation results\n\n"
        ans += f"**Dataset:** {self.dataset.name} \\\n"
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
            caption=f"Confusion matrix for {model_name}. Dataset: {self.dataset.name}",
            model_name=model_name,
            c_cols="c" * len(classes),
            c_line_top=str(len(classes) + 1),
            cls_count=len(classes),
            cls_head=" & ".join(classes),
            cls_rows=cls_rows,
            recalls=" & ".join(
                [f"{str(round(rc * 100, 2))} \\%" for rc in self.precision]
            ),
        )
        return ans
