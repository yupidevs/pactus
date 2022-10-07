from abc import ABC, abstractmethod
from functools import reduce
from typing import Any
from functools import reduce

from yupi import Trajectory
from yuca.dataset import Dataset, DatasetSlice
from yuca.models.evaluation import Evaluation
from __future__ import annotations

def mark_trained(func):
    def wrapper(self: Model, *args, **kwargs):
        val = func(*args, **kwargs)
        self.trained = True
        return val
    return wrapper

class Model(ABC):

    def __new__(cls: type[Self]) -> Self:
        Model.__setattr__(cls.__name__) = cls
        return super().__new__()

    def __init__(self, name: str):
        self.name = name
        self.summary = {"name": self.name}
        self.trained = False

    @abstractmethod
    @mark_trained
    def train(self, data: Dataset | DatasetSlice, cross_validation: int = 0):
        """Train the model using a given dataset"""
    
    @abstractmethod
    def predict(self, traj: Trajectory) -> Any:
        """"""

    def _predict(self, traj: Trajectory) -> Any:
        if not self.trained:
            raise Exception("Model is not trained yet.")
        
        return self.predict(traj)
            
    def evaluate(self, data: Dataset | DatasetSlice) -> Evaluation:
        """Evaluate the trained model"""

        predictions = reduce(self._predict, data.trajs)
        return Evaluation(self.summary, data, predictions)