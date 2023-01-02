import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from yupi import Trajectory

from pactus import config as cfg
from pactus.dataset import Data
from pactus.models.evaluation import Evaluation
from pactus.models.model import Model

NAME = "lstm"

DEFAULT_CALLBACKS = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]


class LSTMModel(Model):
    """Implementation of a LSTM Classifier."""

    def __init__(
        self,
        units: Union[List[int], None] = None,
        masking_value: Union[int, None] = None,
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=None,
        **kwargs,
    ):
        super().__init__(NAME)
        self.masking_value = cfg.MASK_VALUE if masking_value is None else masking_value
        self.encoder: Union[LabelEncoder, None] = None
        self.model: keras.Secuential
        self.max_len = 0
        metrics = ["accuracy"] if metrics is None else metrics
        self.units = [128, 64] if units is None else units
        self.set_summary(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def _get_x_data(self, max_len: int, trajs: List[Trajectory]) -> np.ndarray:
        _X = np.empty(
            (
                len(trajs),
                max_len,
                trajs[0].dim + 1,  # all pos dim plust time
            )
        )
        _X[:, :, :] = self.masking_value
        for i, traj in enumerate(trajs):
            top = len(traj)
            for dim in range(traj.dim):
                _X[i, :top, dim] = traj.r.component(dim)
            _X[i, :top, -1] = traj.t
        return _X

    def _get_model(self, input_shape, n_classes):
        max_len, traj_dim = input_shape
        model = keras.Sequential()
        model.add(
            layers.Masking(
                mask_value=self.masking_value, input_shape=(max_len, traj_dim)
            )
        )
        for units_val in self.units:
            model.add(
                layers.LSTM(
                    units_val,
                    input_shape=(max_len, traj_dim),
                    return_sequences=True,
                )
            )
        model.add(
            layers.Bidirectional(
                layers.LSTM(32, input_shape=(max_len, traj_dim)), merge_mode="ave"
            )
        )
        model.add(layers.Dense(15, activation="relu"))
        model.add(layers.Dense(n_classes, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"],
        )
        return model

    def _prepare_data(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        self.encoder = LabelEncoder()
        self.encoder.fit(data.labels)
        encoded_labels = self.encoder.transform(data.labels)
        y_data = np.array(encoded_labels)

        self.max_len = max(map(len, data.dataset.trajs))
        x_data = self._get_x_data(self.max_len, data.trajs)
        return x_data, y_data

    def train(
        self,
        data: Data,
        cross_validation=0,
        epochs=10,
        batch_size=1,
        validation_split=None,
        callbacks: Union[list, None] = None,
        checkpoint: Union[keras.callbacks.ModelCheckpoint, None] = None,
    ):
        if cross_validation != 0:
            logging.warning("Cross validation is not supported yet for lstm")
        self.set_summary(epochs=epochs, validation_split=validation_split)
        callbacks = DEFAULT_CALLBACKS.copy() if callbacks is None else callbacks
        model_path = None
        if checkpoint is not None:
            callbacks.append(checkpoint)
            if Path(checkpoint.filepath).exists():
                logging.info("Loading model from checkpoint %s", checkpoint.filepath)
                model_path = checkpoint.filepath
        x_train, y_train = self._prepare_data(data)
        self.model = (
            self._get_model(
                input_shape=(
                    x_train.shape[1],  # max len
                    x_train.shape[2],  # trajs dim
                ),
                n_classes=len(data.label_counts),
            )
            if model_path is None
            else keras.models.load_model(model_path)
        )
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
        )

    def predict(self, data: Data) -> List[Any]:
        x_data = self._get_x_data(self.max_len, data.trajs)
        return self.model.predict(x_data)

    def evaluate(self, data: Data) -> Evaluation:
        assert self.encoder is not None, "Encoder is not set."
        x_data = self._get_x_data(self.max_len, data.trajs)
        preds = self.model.predict(x_data)
        preds = [pred.argmax() for pred in preds]
        evals = self.encoder.inverse_transform(preds)
        return Evaluation(self.summary, data, evals)
