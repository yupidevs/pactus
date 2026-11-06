import importlib
from typing import Any

from pactus.models.decision_tree_model import DecisionTreeModel
from pactus.models.evaluation import Evaluation
from pactus.models.evaluation_comparison import EvaluationComparison
from pactus.models.kneighbors_model import KNeighborsModel
from pactus.models.model import Model
from pactus.models.random_forest_model import RandomForestModel
from pactus.models.svm_model import SVMModel
from pactus.models.xgboost_model import XGBoostModel

# These need TensorFlow, so they are only imported when used
_TENSORFLOW_MODELS = {
    "LSTMModel": "pactus.models.lstm_model",
    "TransformerModel": "pactus.models.transformer_model",
}

__all__ = [
    "DecisionTreeModel",
    "Evaluation",
    "EvaluationComparison",
    "KNeighborsModel",
    "LSTMModel",
    "Model",
    "RandomForestModel",
    "SVMModel",
    "TransformerModel",
    "XGBoostModel",
]


def __getattr__(name: str) -> Any:
    if name in _TENSORFLOW_MODELS:
        return getattr(importlib.import_module(_TENSORFLOW_MODELS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
