from re import S

import numpy as np
import pytest
import yupi
from yupi.core.featurizers import SpatialFeaturizer

from pactus.dataset import Data

np.random.seed(42)


@pytest.fixture
def data():
    trajs = [
        yupi.Trajectory(x=np.random.rand(100), y=np.random.rand(100))
        for _ in range(1000)
    ]
    labels = np.random.choice(["a", "b", "c"], size=1000)
    return Data(trajs, labels)


def test_data(data: Data):
    assert len(data) == 1000
    assert len(data.labels) == 1000
    assert len(data.trajs) == 1000
    assert set(data.classes) == {"a", "b", "c"}


def test_data_float_cut(data: Data):
    left, right = data.cut(0.5)
    assert len(left) == 500
    assert len(right) == 500

    with pytest.raises(AssertionError):
        data.cut(1.5)

    with pytest.raises(AssertionError):
        data.cut(-0.5)


def test_data_int_cut(data: Data):
    left, right = data.cut(500)
    assert len(left) == 500
    assert len(right) == 500

    with pytest.raises(AssertionError):
        data.cut(1500)

    with pytest.raises(AssertionError):
        data.cut(-500)


def test_data_float_split(data: Data):
    train, test = data.split(0.8)
    assert len(train) == 800
    assert len(test) == 200

    with pytest.raises(AssertionError):
        data.split(1.5)

    with pytest.raises(AssertionError):
        data.split(-0.5)


def test_data_int_split(data: Data):
    train, test = data.split(800)
    assert len(train) == 800
    assert len(test) == 200

    with pytest.raises(AssertionError):
        data.split(1500)

    with pytest.raises(AssertionError):
        data.split(-500)

def test_data_float_take(data: Data):
    sub_data = data.take(0.8)
    assert len(sub_data) == 800

    with pytest.raises(AssertionError):
        data.take(1.5)

    with pytest.raises(AssertionError):
        data.take(-0.5)

def test_data_int_take(data: Data):
    sub_data = data.take(800)
    assert len(sub_data) == 800

    with pytest.raises(AssertionError):
        data.take(1500)

    with pytest.raises(AssertionError):
        data.take(-500)

def test_data_featurize(data: Data):
    featurizer = SpatialFeaturizer()
    featurized_data = data.featurize(featurizer)
    assert len(featurized_data) == 1000
    assert len(featurized_data[0]) == featurizer.count