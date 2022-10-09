import logging

from yupi import DiffMethod, Trajectory, WindowType

from yuca.dataset import Dataset, GeoLifeDataset, LangevinDataset
from yuca.features import Featurizer
from yuca.models import Model, RandomForestModel

__version__ = "0.1.0"

Trajectory.global_diff_method(
    method=DiffMethod.LINEAR_DIFF, window_type=WindowType.FORWARD
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
