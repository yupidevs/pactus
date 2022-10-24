from string import Template
from typing import Any, Dict, List

from pactus.models.evaluation import Evaluation

LATEX_EVAL_ROW_TEMPLATE = Template(
    r"""	   \multirow{2}{*}{$model_name}
			   & ACC & $acc_vals \\
			   & F-Score & $f_score_vals\\
"""
)

LATEX_EVAL_TEMPLATE = Template(
    r"""
\begin{figure}[ht!]
\caption{$caption}
\vspace{2mm}
\centering
   \begin{tabular}{cr|$ds_cols_chars}
	   \toprule
	   \multirow{2}[4]{*}{\textbf{Model}} &
	   \multirow{2}[4]{*}{\textbf{Metrics}} &
	   \multicolumn{$ds_count}{c}{\textbf{Datasets}} \\
       & \cline{2-$c_line_top}
	   & & $ds_names \\
	   \midrule
$rows
	   \bottomrule
   \end{tabular}
\end{figure}
"""
)


class EvaluationComparison:
    def __init__(self, evals: List[Evaluation]):
        self.evals = evals

        self.evals_by_model: Dict[str, List[Evaluation]] = {}
        for evaluation in self.evals:
            model = evaluation.model_summary["name"]
            self.evals_by_model[model] = self.evals_by_model.get(model, []) + [
                evaluation
            ]

        self.evals_by_dataset: Dict[str, List[Evaluation]] = {}
        for evaluation in self.evals:
            ds_name = evaluation.dataset.name
            self.evals_by_dataset[ds_name] = self.evals_by_dataset.get(ds_name, []) + [
                evaluation
            ]

    def to_latex(self) -> str:
        rows = []
        for model, evaluations in self.evals_by_model.items():
            model_name = model.replace("_", " ").title()
            rows.append(
                LATEX_EVAL_ROW_TEMPLATE.substitute(
                    model_name=model_name,
                    acc_vals=" & ".join(
                        [f"{e.acc_overall:.2f}\\%" for e in evaluations]
                    ),
                    f_score_vals=" & ".join(
                        [f"{e.f1_score:.2f}\\%" for e in evaluations]
                    ),
                )
            )
        ds_names = [ds.replace("_", " ").title() for ds in self.evals_by_dataset]
        return LATEX_EVAL_TEMPLATE.substitute(
            caption="Evaluation Comparison",
            ds_cols_chars="c" * len(self.evals_by_dataset),
            c_line_top=len(self.evals_by_dataset) + 1,
            ds_names=" & ".join(ds_names),
            ds_count=len(self.evals_by_dataset),
            rows="".join(rows),
        )
