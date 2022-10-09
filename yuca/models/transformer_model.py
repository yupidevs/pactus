import numpy as np
from typing import Any

from yupi import Trajectory

from yuca.dataset import Data
from yuca.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from yuca.models.transformer import build_model

NAME = "transformer_model"


class TransformerModel(Model):
    """Implementation of a Transformer model."""

    def __init__(
        self,
        head_size=256,
        num_heads=1,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=None,
        mlp_droput=0.4,
        droput=0.25,
        loss="categorical_crossentropy",
        optimizer=None,
        metrics=None
    ):
        super().__init__(NAME)
        self.kwargs = kwargs
        self.head_size=head_size,
        self.num_heads=num_heads,
        self.ff_dim=ff_dim,
        self.num_transformer_blocks=num_transformer_blocks,
        self.mlp_units=[128] if mlp_units is None else mlp_units,
        self.mlp_dropout=mlp_droput,
        self.dropout=droput,
        self.model: keras.Model
        self.loss = loss
        self.optimizer = keras.optimizer.Adam(learning_rate=1e-2) if optimizer is None else optimizer
        self.metrics = ["accuracy"] if metrics is None else metrics

    def train(self, data: Data, cross_validation: int = 0):
        x_train, y_train = self._get_input_data(data)
        n_classes = len(data.dataset.classes)
        self.model = tr.build_model(
            n_classes,
            input_shape,
            input_mask=mask,
        )
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )

    def predict(self, data: Data) -> list[Any]:
        raise NotImplementedError

    def _get_input_data(self, data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process all the data and returns a x_data, y_data, mask readable
        by the transformer
        """
        y_data = self._encode_labels(data)
        x_data = self._extract_raw_data(data)
        x_data = self._reshape_input(x_data)
        # mask = self._mask_data(x_data)
        return x_data, y_data, None
    
    def _mask_data(self, x_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _encode_labels(self, labels: data: Data) -> np.ndarray:
        """Encode the labels"""
        encoder = LabelEncoder()
        encoder.fit(data.dataset.lables)
        encoded_labels = encoder.transform(data.labels)
        assert isinstance(encoded_labels, np.ndarray)

        classes = np.zeros((len(encoded_labels), len(encoder.classes_)))
        for i, label in enumerate(encoded_labels):
            classes[i][label] = 1
        return classes

    def _extract_raw_data(self, data: Data) -> np.ndarray:
        """Extracts the raw data from the yupi trajectories"""
        trajs = data.trajs
        max_len = np.max([len(traj) for traj in data.dataset.trajs])
        raw_data = [np.hstack((traj.r, np.reshape(traj.t, (-1, 1)))) for traj in trajs]
        all_raw_data = np.zeros((len(raw_data), max_len, 3))
        for i, traj in enumerate(raw_data):
            all_raw_data[i, :, :] = 0  # TODO: check for masking
            all_raw_data[i, : len(traj)] = traj
        return all_raw_data
    
    def reshape_input(self, x_data: np.ndarray) -> np.ndarray:
        """Reshapes the input data to be compatible with the transformer."""
        return x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))