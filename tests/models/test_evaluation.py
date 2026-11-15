from pathlib import Path

import numpy as np
import pytest

from pactus import Evaluation, EvaluationComparison


def _evaluation(name: str = "my_model", dataset: str = "ds") -> Evaluation:
    return Evaluation(
        dataset_name=dataset,
        trajs_ids=["0", "1", "2", "3"],
        y_true=["a", "a", "b", "b"],
        y_pred=np.array(["a", "b", "b", "b"]),
        model_summary={"name": name, "depth": 2},
    )


def test_metrics() -> None:
    evaluation = _evaluation()
    assert evaluation.acc_overall == pytest.approx(0.75)
    assert evaluation.classes == ["a", "b"]
    assert evaluation.precision[0] == pytest.approx(1.0)
    assert evaluation.recall[0] == pytest.approx(0.5)


def test_show_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    _evaluation().show()
    out = capsys.readouterr().out
    assert "Accuracy: 0.750" in out
    assert "Confusion matrix" in out


def test_markdown_report() -> None:
    assert "| **a** |" in _evaluation().to_markdown()


def test_latex_report() -> None:
    assert "Predicted" in _evaluation().to_latex()


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "evaluation.json"
    _evaluation().save(str(path))
    loaded = Evaluation.load(str(path))
    assert loaded.y_pred == ["a", "b", "b", "b"]
    assert loaded.model_summary["name"] == "my_model"


def test_save_needs_one_id_per_prediction(tmp_path: Path) -> None:
    evaluation = _evaluation()
    evaluation.traj_ids = ["0"]
    with pytest.raises(ValueError, match="trajectory id"):
        evaluation.save(str(tmp_path / "evaluation.json"))


def test_load_needs_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="json"):
        Evaluation.load(str(tmp_path / "evaluation.txt"))


def test_comparison_table() -> None:
    latex = EvaluationComparison(
        [
            _evaluation("first_model", "one"),
            _evaluation("first_model", "two"),
            _evaluation("other", "one"),
        ]
    ).to_latex()
    assert "First Model" in latex
    assert "One & Two" in latex
