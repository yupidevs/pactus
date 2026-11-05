from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pactus.dataset import Data
from pactus.models.evaluation import Evaluation

logger = logging.getLogger(__name__)


def _mark_trained(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: Model, *args: Any, **kwargs: Any) -> Any:
        logger.info("Training %s model", self.name)
        val = func(self, *args, **kwargs)
        self.trained = True
        return val

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Model(ABC):
    """Abstract class for models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.summary = {"name": name}
        self.trained = False

    @abstractmethod
    @_mark_trained
    def train(self, data: Data, cross_validation: int = 0) -> None:
        """Train the model using a given dataset"""

    @abstractmethod
    def predict(self, data: Data) -> list[Any]:
        """Predict the labels of a given set of trajectories"""

    def set_summary(self, **summary: Any) -> None:
        """Set the summary of the model"""
        self.summary.update(summary)

    def save(self, path: str) -> None:
        """Save the model to a given path"""
        raise NotImplementedError

    def _predict(self, data: Data) -> list[Any]:
        return self.predict(data)

    def evaluate(self, data: Data) -> Evaluation:
        """Evaluate the trained model"""

        logger.info("Evaluating the %s model", self.name)
        predictions = self._predict(data)
        return Evaluation.from_data(data, predictions, self.summary)
