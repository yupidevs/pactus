import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from yupi import Trajectory

import yuca.transformer as tr
from yuca.extra_pl_steps import dataset_description
from yuca.ml_pipeline import get_train_test_splitter
from yuca.pipeline import Pipeline, PipelineStep

TrainTestTuples = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@PipelineStep.build("class encoder")
def class_encoder(
    raw_data: list[Trajectory], clss: np.ndarray
) -> tuple[list[Trajectory], np.ndarray]:
    """
    Encodes the classes into numbers.

    Input: list of feature vectors, classes
    Output: tuple of two elements (feature vectors, encoded classes)
    """
    encoder = LabelEncoder()
    encoder.fit(clss)
    encoded_clss = encoder.transform(clss)
    assert isinstance(encoded_clss, np.ndarray)

    classes = np.zeros((len(encoded_clss), len(encoder.classes_)))
    for i, label in enumerate(encoded_clss):
        classes[i][label] = 1
    return raw_data, classes


@PipelineStep.build("raw data extractor")
def raw_data_extractor(
    trajs: list[Trajectory], classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw_data = [np.hstack((traj.r, np.reshape(traj.t, (-1, 1)))) for traj in trajs]
    max_len = np.max([len(traj) for traj in raw_data])
    all_raw_data = np.zeros((len(raw_data), max_len, 3))
    for i, traj in enumerate(raw_data):
        all_raw_data[i, :, :] = 0  # TODO: check for masking
        all_raw_data[i, : len(traj)] = traj

    return all_raw_data, classes


@PipelineStep.build("reshape input")
def reshape_input(splitted_data: TrainTestTuples) -> TrainTestTuples:
    """
    Reshapes the input data to be compatible with the transformer.

    Input: tuple of np.ndarray (X_train, y_train, X_test, y_test)
    Output: tuple of np.ndarray (X_train, y_train, X_test, y_test)
    """
    x_train, y_train, x_test, y_test = splitted_data
    train_input_shape = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    test_input_shape = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train.reshape(train_input_shape)
    x_test = x_test.reshape(test_input_shape)
    return x_train, y_train, x_test, y_test


def _get_input_mask(x_data: np.ndarray) -> np.ndarray:
    mask = np.ones(x_data.shape)
    mask[np.isnan(x_data)] = 0
    return np.array(mask, dtype=bool)


def get_transformer_classifier_pl(
    test_size: float = 0.2, random_state: int | None = None
) -> Pipeline:
    """
    Creates a pipeline that trains a transformer model.

    Returns
    -------
    Pipeline
        The pipeline.
    """

    @PipelineStep.build("transformer")
    def transformer_classifier(
        splitted_data: TrainTestTuples,
    ) -> None:
        """
        Takes the splitted data and trains a transformer model.

        Input: tuple of np.ndarray (X_train, y_train, X_test, y_test)
        Output: None
        """
        x_train, y_train, x_test, y_test = splitted_data

        input_shape = x_train.shape[1:]

        mask = None
        # mask = _get_input_mask(x_train)

        n_classes = len(np.unique(y_train))
        model = tr.build_model(
            n_classes,
            input_shape,
            head_size=256,
            num_heads=1,
            ff_dim=4,
            num_transformer_blocks=2,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
            input_mask=mask,
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            metrics=["accuracy"],
        )
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]

        model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=callbacks,
        )

        model.evaluate(x_test, y_test, verbose=2)

    return Pipeline(
        "transformer classifier",
        dataset_description,
        class_encoder,
        raw_data_extractor,
        get_train_test_splitter(test_size=test_size, random_state=random_state),
        reshape_input,
        transformer_classifier,
    )
