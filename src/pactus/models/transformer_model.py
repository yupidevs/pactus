import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import pactus.config as cfg
from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation
from pactus.models.transformer import TransformerBlock, build_model

logger = logging.getLogger(__name__)

NAME = "transformer_model"
DEFAULT_OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-2)
DEFAULT_CALLBACKS = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]


class TransformerModel(Model):
    """Implementation of a Transformer model."""

    def __init__(
        self,
        head_size: int = 256,
        num_heads: int = 1,
        ff_dim: int = 4,
        num_transformer_blocks: int = 2,
        mlp_units: list[int] | None = None,
        mlp_dropout: float = 0.4,
        dropout: float = 0.25,
        loss: str = "categorical_crossentropy",
        optimizer: Any = None,
        metrics: list[Any] | None = None,
        max_traj_len: int = -1,
        skip_long_trajs: bool = False,
        mask_value: int = cfg.MASK_VALUE,
        random_state: int | None = None,
    ) -> None:
        super().__init__(NAME)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = [128] if mlp_units is None else mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.model: keras.Model
        self.loss = loss
        self.optimizer = DEFAULT_OPTIMIZER if optimizer is None else optimizer
        self.metrics = ["accuracy"] if metrics is None else metrics
        self.max_traj_len = max_traj_len
        self.skip_long_trajs = skip_long_trajs
        self.mask_value = mask_value
        self.encoder: LabelEncoder | None = None
        self.labels: list[Any] | None = None
        self.original_data: Data | None = None
        self.random_state: int | None = random_state
        self.set_summary(
            head_size=self.head_size,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=self.mlp_units,
            mlp_dropout=self.mlp_dropout,
            dropout=self.dropout,
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            max_traj_len=self.max_traj_len,
            skip_long_trajs=self.skip_long_trajs,
        )

    def train(
        self,
        data: Data,
        original_data: Data,
        cross_validation: int = 0,
        epochs: int = 10,
        validation_split: float = 0.2,
        batch_size: int = 32,
        callbacks: list | None = None,
        checkpoint: keras.callbacks.ModelCheckpoint | None = None,
    ) -> None:
        if self.random_state is not None:
            tf.keras.utils.set_random_seed(self.random_state)
            logger.warning(
                "Custom seed provided for %s model. This calls "
                "'tf.keras.utils.set_random_seed' which sets a global "
                "random state on python, numpy and tensorflow.",
                self.name,
            )
        self.set_summary(
            cross_validation=cross_validation,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
        )
        self.encoder = None
        self.labels = data.labels
        self.original_data = original_data
        x_train, y_train = self._get_input_data(data)
        n_classes = len(data.classes)
        input_shape = x_train.shape[1:]
        callbacks = DEFAULT_CALLBACKS.copy() if callbacks is None else callbacks
        model_path = None
        if checkpoint is not None:
            callbacks.append(checkpoint)
            if Path(checkpoint.filepath).exists():
                logger.info("Loading model from checkpoint %s", checkpoint.filepath)
                model_path = checkpoint.filepath

        if cross_validation == 0:
            model = (
                self._get_model(n_classes, input_shape)
                if model_path is None
                else keras.models.load_model(
                    model_path, custom_objects={"TransformerBlock": TransformerBlock}
                )
            )
            model.fit(
                x_train,
                y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
            )
            self.model = model
        else:
            if not (cross_validation > 1):
                raise ValueError("cross_validation must be greater than 1")
            kfold = KFold(n_splits=cross_validation, shuffle=True)

            best_acc = -1
            fold_no = 1
            for train_idxs, test_idxs in kfold.split(x_train, y_train):
                x_train_fold = x_train[train_idxs]
                y_train_fold = y_train[train_idxs]
                model = self._get_model(n_classes, input_shape)
                model.fit(
                    x_train_fold,
                    y_train_fold,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                )

                x_test_fold = x_train[test_idxs]
                y_test_fold = y_train[test_idxs]
                scores = model.evaluate(x_test_fold, y_test_fold, verbose=2)
                acc = scores[1]
                loss = scores[0]

                logger.info("Fold %d: Loss: %f, Accuracy: %f", fold_no, loss, acc)

                if acc > best_acc:
                    self.model = model
                fold_no += 1

    def predict(self, data: Data) -> list[Any]:
        x_data, _ = self._get_input_data(data)
        predictions: list[Any] = self.model.predict(x_data)
        return predictions

    def _get_model(self, n_classes: int, input_shape: tuple) -> keras.Model:
        model = build_model(
            n_classes,
            input_shape=input_shape,
            head_size=self.head_size,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=self.mlp_units,
            mlp_dropout=self.mlp_dropout,
            dropout=self.dropout,
        )
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model

    def _get_input_data(self, data: Data) -> tuple[np.ndarray, np.ndarray]:
        """
        Process all the data and returns a x_data, y_data, mask readable
        by the transformer
        """
        y_data = self._encode_labels(data)
        x_data = self._extract_raw_data(data)
        x_data = self._reshape_input(x_data)
        return x_data, y_data

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels"""
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(self.labels)
        encoded_labels = self.encoder.transform(data.labels)

        classes = np.zeros((len(encoded_labels), len(self.encoder.classes_)))
        for i, label in enumerate(encoded_labels):
            classes[i][label] = 1
        return classes

    def _extract_raw_data(self, data: Data) -> np.ndarray:
        """Extracts the raw data from the yupi trajectories"""
        if not (self.original_data is not None):
            raise ValueError("Original data must be set")

        trajs = data.trajs
        max_len = np.max([len(traj) for traj in self.original_data.trajs])
        if self.max_traj_len > 0:
            max_len = self.max_traj_len
        raw_data = [np.hstack((traj.r, np.reshape(traj.t, (-1, 1)))) for traj in trajs]
        if self.skip_long_trajs:
            raw_data = [traj for traj in raw_data if traj.shape[0] <= max_len]
        if not (len(raw_data) > 0):
            raise ValueError("No trajectories to train on")
        all_raw_data = np.zeros((len(raw_data), max_len, 3))
        for i, raw_traj in enumerate(raw_data):
            traj = raw_traj[:max_len]
            all_raw_data[i, :, :] = self.mask_value
            all_raw_data[i, : traj.shape[0]] = traj
        return all_raw_data

    def _reshape_input(self, x_data: np.ndarray) -> np.ndarray:
        """Reshapes the input data to be compatible with the transformer."""
        return x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))

    def evaluate(self, data: Data) -> Evaluation:
        if not (self.encoder is not None):
            raise ValueError("Encoder is not set.")
        x_data, _ = self._get_input_data(data)
        preds = self.model.predict(x_data)
        preds = [pred.argmax() for pred in preds]
        evals = self.encoder.inverse_transform(preds)
        return Evaluation.from_data(data, evals, self.summary)
