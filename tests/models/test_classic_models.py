from typing import Any

import pytest

from pactus import Dataset, Evaluation, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    Model,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)

MODELS: list[tuple[Any, dict[str, Any]]] = [
    (RandomForestModel, {"n_estimators": 5, "random_state": 0}),
    (DecisionTreeModel, {"random_state": 0}),
    (KNeighborsModel, {"n_neighbors": 1}),
    (SVMModel, {}),
    (XGBoostModel, {"n_estimators": 3, "random_state": 0}),
]


@pytest.mark.parametrize(
    ("model_class", "options"), MODELS, ids=lambda v: getattr(v, "__name__", "")
)
def test_models_train_and_evaluate(
    dataset: Dataset, model_class: Any, options: dict[str, Any]
) -> None:
    train, test = dataset.split(0.75, random_state=0)
    model = model_class(featurizer=featurizers.SpatialFeaturizer(), **options)
    model.train(train, cross_validation=2)
    evaluation = model.evaluate(test)
    assert isinstance(evaluation, Evaluation)
    assert len(evaluation.y_pred) == len(test)
    assert evaluation.model_summary["cross_validation"] == 2


@pytest.mark.parametrize(
    ("model_class", "options"), MODELS, ids=lambda v: getattr(v, "__name__", "")
)
def test_a_single_trajectory_can_be_predicted(
    dataset: Dataset, model_class: Any, options: dict[str, Any]
) -> None:
    model = model_class(featurizer=featurizers.SpatialFeaturizer(), **options)
    model.train(dataset, cross_validation=2)
    assert model.predict_single(dataset.trajs[0]) in dataset.classes


def test_saving_is_not_available_by_default() -> None:
    class Empty(Model):
        def train(self, data: Any, cross_validation: int = 0) -> None: ...

        def predict(self, data: Any) -> list[Any]:
            return []

    with pytest.raises(NotImplementedError):
        Empty("empty").save("anywhere")
