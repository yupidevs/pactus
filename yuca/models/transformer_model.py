from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from yuca.dataset import Data
from yuca.models import Model
from yuca.models.transformer import build_model

NAME = "transformer_model"
DEFAULT_OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-2)
DEFAULT_CALLBACKS = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]


class TransformerModel(Model):
    """Implementation of a Transformer model."""

    def __init__(
        self,
        head_size: int = 256,
        num_heads: int = 1,
        ff_dim: int = 4,
        num_transformer_blocks: int = 2,
        mlp_units: list[int] | None = None,
        mlp_droput: float = 0.4,
        droput: float = 0.25,
        loss="categorical_crossentropy",
        optimizer=None,
        metrics=None,
    ):
        super().__init__(NAME)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = [128] if mlp_units is None else mlp_units
        self.mlp_dropout = mlp_droput
        self.dropout = droput
        self.model: keras.Model
        self.loss = loss
        self.optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer
        self.metrics = ["accuracy"] if metrics is None else metrics

    def train(
        self,
        data: Data,
        cross_validation: int = 0,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: list | None = None,
    ):
        x_train, y_train, mask = self._get_input_data(data)
        n_classes = len(data.dataset.classes)
        input_shape = x_train.shape[1:]
        callbacks = DEFAULT_CALLBACKS if callbacks is None else callbacks

        self.model = (
            build_model(
                n_classes,
                input_shape,
                input_mask=mask,
                head_size=self.head_size,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                num_transformer_blocks=self.num_transformer_blocks,
                mlp_units=self.mlp_units,
                mlp_dropout=self.mlp_dropout,
                dropout=self.dropout,
            )
            .compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics,
            )
            .fit(
                x_train,
                y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
            )
        )

    def predict(self, data: Data) -> list[Any]:
        x_data, _, _ = self._get_input_data(data)
        return self.model.predict(x_data)

    def _get_input_data(
        self, data: Data
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels"""
        encoder = LabelEncoder()
        encoder.fit(data.dataset.labels)
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

    def _reshape_input(self, x_data: np.ndarray) -> np.ndarray:
        """Reshapes the input data to be compatible with the transformer."""
        x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        return x_data
