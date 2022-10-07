from typing import Any

from yupi import Trajectory
from yuca.dataset import Dataset, DatasetSlice
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from yuca.features.featurizer import Featurizer
from yuca.models import Model

NAME = "random_forest"

class RandomForestModel(Model):

    def __init__(self, featurizer: Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.rfc = RandomForestClassifier(**kwargs)
    
    def train(self, data: Dataset | DatasetSlice, cross_validation: int = 0):
        rfc = GridSearchCV(rfc, {}, cv=cross_validation, verbose=3)
        x_data = self.featurizer.compute(data)
        self.rfc.fit(x_data, data.labels)
    
    def predict(self, trajs: list[Trajectory]) -> list[Any]:
        x_data = self.featurizer.compute(trajs)
        return self.rfc.predict(x_data)

    def predict_single(self, traj: Trajectory) -> Any:
        return self.predict([traj])[0]
