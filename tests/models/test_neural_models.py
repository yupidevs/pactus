import numpy as np
import pytest

from pactus import Dataset, Evaluation
from pactus.models import LSTMModel, TransformerModel
from pactus.models.transformer import TransformerBlock


def _transformer(**options: object) -> TransformerModel:
    return TransformerModel(
        head_size=4,
        ff_dim=2,
        num_transformer_blocks=1,
        mlp_units=[4],
        optimizer="adam",
        random_state=0,
        **options,  # type: ignore[arg-type]
    )


def test_lstm_trains_and_evaluates(dataset: Dataset) -> None:
    train, test = dataset.split(0.75, random_state=0)
    model = LSTMModel(units=[2], random_state=0)
    model.train(train, dataset, epochs=1, batch_size=8)
    evaluation = model.evaluate(test)
    assert isinstance(evaluation, Evaluation)
    assert len(evaluation.y_pred) == len(test)
    assert np.asarray(model.predict(test)).shape == (len(test), 2)


def test_transformer_trains_and_evaluates(dataset: Dataset) -> None:
    train, test = dataset.split(0.75, random_state=0)
    model = _transformer()
    model.train(train, dataset, epochs=1, batch_size=8)
    assert isinstance(model.evaluate(test), Evaluation)
    assert np.asarray(model.predict(test)).shape == (len(test), 2)


def test_transformer_with_cross_validation(dataset: Dataset) -> None:
    train, test = dataset.split(0.75, random_state=0)
    model = _transformer()
    model.train(train, dataset, cross_validation=2, epochs=1, batch_size=8)
    assert isinstance(model.evaluate(test), Evaluation)


def test_transformer_skips_long_trajectories(dataset: Dataset) -> None:
    train, _ = dataset.split(0.75, random_state=0)
    model = _transformer(max_traj_len=5, skip_long_trajs=True)
    model.encoder = None
    model.labels = train.labels
    model.original_data = train
    with pytest.raises(ValueError, match="No trajectories"):
        model._extract_raw_data(train)


def test_transformer_needs_its_training_data(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="Original data"):
        _transformer()._extract_raw_data(dataset)


def test_transformer_cross_validation_needs_two_folds(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        _transformer().train(dataset, dataset, cross_validation=1, epochs=1)


def test_evaluating_before_training_is_an_error(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="Encoder is not set"):
        _transformer().evaluate(dataset)
    with pytest.raises(ValueError, match="Encoder is not set"):
        LSTMModel().evaluate(dataset)


def test_transformer_block_config_round_trip() -> None:
    block = TransformerBlock(4, 1, 2, 1, rate=0.3)
    config = block.get_config()
    assert (config["head_size"], config["ff_dim2"], config["rate"]) == (4, 1, 0.3)
