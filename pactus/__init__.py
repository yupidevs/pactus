import logging

from yupi import DiffMethod, Trajectory, WindowType

from pactus.dataset import Dataset
from pactus.models import (
    DecisionTreeModel,
    Evaluation,
    EvaluationComparison,
    KNeighborsModel,
    Model,
    RandomForestModel,
    SVMModel,
    TransformerModel,
)

__version__ = "0.1.1a1"

Trajectory.global_diff_method(
    method=DiffMethod.LINEAR_DIFF, window_type=WindowType.FORWARD
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
