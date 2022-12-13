import logging
from collections import Counter
from pathlib import Path
from typing import Any, List, Union

from tensorflow import keras

import pactus.config as cfg
from pactus.dataset import Data
from pactus.models.evaluation import Evaluation
from pactus.models.transformer_model import TransformerModel
from pactus.models.zoomletizer import zoomletize

NAME = "zoom_transformer_model"
DEFAULT_OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-2)
DEFAULT_CALLBACKS = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]


class ZoomTransformerModel(TransformerModel):
    """Implementation of a Zoom Transformer model."""

    def __init__(
        self,
        head_size: int = 256,
        num_heads: int = 1,
        ff_dim: int = 4,
        num_transformer_blocks: int = 2,
        mlp_units: Union[List[int], None] = None,
        mlp_dropout: float = 0.4,
        dropout: float = 0.25,
        loss="categorical_crossentropy",
        optimizer=None,
        metrics=None,
        max_traj_len: int = -1,
        skip_long_trajs: bool = False,
        mask_value=cfg.MASK_VALUE,
        zoomlet_size: int = 30,
        zoomlet_zoom: Any = -1,
        zoomlet_shift: int = -1,
    ):
        super().__init__(
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            mlp_dropout=mlp_dropout,
            dropout=dropout,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            max_traj_len=max_traj_len,
            skip_long_trajs=skip_long_trajs,
            mask_value=mask_value,
            name=NAME,
        )
        self.zoomlet_size = zoomlet_size
        self.zoomlet_zoom = zoomlet_zoom
        self.zoomlet_shift = zoomlet_shift

    def train(
        self,
        data: Data,
        cross_validation: int = 0,
        epochs: int = 10,
        validation_split: float = 0.2,
        batch_size: int = 32,
        callbacks: Union[list, None] = None,
        checkpoint: Union[keras.callbacks.ModelCheckpoint, None] = None,
    ):
        self.set_summary(
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
        )
        self.encoder = None
        self.labels = data.dataset.labels

        if cross_validation > 0:
            logging.warning("Cross validation is not implemented for this model.")

        callbacks = DEFAULT_CALLBACKS if callbacks is None else callbacks
        model_path = None
        if checkpoint is not None:
            callbacks.append(checkpoint)
            if Path(checkpoint.filepath).exists():
                logging.info("Loading model from checkpoint %s", checkpoint.filepath)
                model_path = checkpoint.filepath

        model = None
        for epoch in range(epochs):
            logging.info("Epoch %d", epoch + 1)
            e_data = zoomletize(
                data,
                zoom=self.zoomlet_zoom,
                shift=self.zoomlet_shift,
                size=self.zoomlet_size,
            )
            x_train, y_train = self._get_input_data(e_data)
            n_classes = len(data.dataset.classes)
            input_shape = x_train.shape[1:]

            if model is None:
                model = (
                    self._get_model(
                        n_classes,
                        input_shape,
                        mask=self.mask_value,
                    )
                    if model_path is None
                    else keras.models.load_model(model_path)
                )

            model.fit(
                x_train,
                y_train,
                validation_split=validation_split,
                epochs=1,
                batch_size=batch_size,
                callbacks=callbacks,
            )
            self.model = model

    def evaluate(self, data: Data) -> Evaluation:
        assert self.encoder is not None, "Encoder is not set."
        evals = []
        i = 0
        for traj, label in zip(data.trajs, data.labels):
            print(f"Eval progress: {i}/{len(data.trajs)}", end="\r")
            i += 1
            new_data = Data(data.dataset, [traj] * 50, [label] * 50)
            e_data = zoomletize(
                new_data,
                zoom=self.zoomlet_zoom,
                shift=self.zoomlet_shift,
                size=self.zoomlet_size,
            )
            x_data, _ = self._get_input_data(e_data)
            preds = self.model.predict(x_data, verbose=0)
            preds = [pred.argmax() for pred in preds]
            counter = Counter(preds)
            most_common = counter.most_common(1)[0][0]
            lbl = self.encoder.inverse_transform([most_common])
            evals.append(lbl)
        return Evaluation(self.summary, data, evals)
