from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from yupi import Trajectory

from yuca.dataset import Dataset, DatasetSlice
from yuca.features.featurizer import Featurizer
from yuca.models import Model

NAME = "random_forest"


class RandomForestModel(Model):
    """Implementation of a Random Forest Classifier."""

    def __init__(self, featurizer: Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.kwargs = kwargs
        self.rfc = RandomForestClassifier(**kwargs)
        self.grid: GridSearchCV

    def train(self, data: Dataset | DatasetSlice, cross_validation: int = 0):
        x_data = self.featurizer.compute(data) if not isinstance(data, list) else data
        self.grid = GridSearchCV(self.rfc, {}, cv=cross_validation, verbose=3)
        self.grid.fit(x_data, data.labels)

    def predict(self, data: Dataset | DatasetSlice) -> list[Any]:
        x_data = self.featurizer.compute(data)
        return self.grid.predict(x_data)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.grid.predict([traj])[0]
