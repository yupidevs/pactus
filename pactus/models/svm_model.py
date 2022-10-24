from typing import Any, List

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from yupi import Trajectory

from pactus.dataset import Data
from pactus.features.featurizer import Featurizer
from pactus.models.model import Model

NAME = "SVM"


class SVMModel(Model):
    """Implementation of a Support Vector Machine Classifier."""

    def __init__(self, featurizer: Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.model = SVC(**kwargs)
        self.grid: GridSearchCV
        self.set_summary(**kwargs)

    def train(self, data: Data, cross_validation: int = 0):
        self.set_summary(cross_validation=cross_validation)
        x_data = self.featurizer.compute(data)
        self.grid = GridSearchCV(self.model, {}, cv=cross_validation, verbose=3)
        self.grid.fit(x_data, data.labels)

    def predict(self, data: Data) -> List[Any]:
        x_data = self.featurizer.compute(data)
        return self.grid.predict(x_data)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.grid.predict([traj])[0]
