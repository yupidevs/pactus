from typing import Any, List

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from yupi import Trajectory

from pactus import featurizers
from pactus.dataset import Data
from pactus.models.model import Model

NAME = "decision_tree"


class DecisionTreeModel(Model):
    """Implementation of a Decision Tree Classifier."""

    def __init__(self, featurizer: featurizers.Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.model = DecisionTreeClassifier(**kwargs)
        self.grid: GridSearchCV
        self.set_summary(**kwargs)

    def train(self, data: Data, cross_validation: int = 0):
        self.set_summary(cross_validation=cross_validation)
        x_data = self.featurizer.featurize(data.trajs)
        self.grid = GridSearchCV(self.model, {}, cv=cross_validation, verbose=3)
        self.grid.fit(x_data, data.labels)

    def predict(self, data: Data) -> List[Any]:
        x_data = self.featurizer.featurize(data.trajs)
        return self.grid.predict(x_data)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.grid.predict([traj])[0]
