import importlib
import sys

import pytest

import pactus.models

NEURAL = (
    "pactus.models.lstm_model",
    "pactus.models.transformer_model",
    "pactus.models.transformer",
)


def test_the_classic_models_do_not_load_tensorflow(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in NEURAL:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(pactus.models)
    assert pactus.models.RandomForestModel is not None
    assert not any(module in sys.modules for module in NEURAL)


@pytest.mark.parametrize("name", ["LSTMModel", "TransformerModel"])
def test_the_neural_models_are_still_importable(name: str) -> None:
    assert getattr(pactus.models, name).__name__ == name
    assert name in pactus.models.__all__


def test_unknown_names_are_errors() -> None:
    with pytest.raises(AttributeError, match="nope"):
        _ = pactus.models.nope
