import logging

from yuca.dataset import Dataset, GeoLifeDataset, LangevinDataset
from yuca.features import Featurizer
from yuca.models import Model, RandomForestModel

__version__ = "0.1.0"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
