import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import yupi
from yupi.core import JSONSerializer
from yupi.core.featurizers import SpatialFeaturizer

from pactus import config
from pactus.dataset import Data, Dataset
from pactus.dataset import dataset as dataset_module

np.random.seed(42)


@pytest.fixture
def data() -> Data:
    trajs = [yupi.Trajectory(x=np.random.rand(100), y=np.random.rand(100)) for _ in range(1000)]
    labels = np.random.choice(["a", "b", "c"], size=1000).tolist()
    return Data(trajs, labels)


def test_data(data: Data) -> None:
    assert len(data) == 1000
    assert len(data.labels) == 1000
    assert len(data.trajs) == 1000
    assert set(data.classes) == {"a", "b", "c"}


def test_data_float_cut(data: Data) -> None:
    left, right = data.cut(0.5)
    assert len(left) == 500
    assert len(right) == 500

    with pytest.raises(ValueError):
        data.cut(1.5)

    with pytest.raises(ValueError):
        data.cut(-0.5)


def test_data_int_cut(data: Data) -> None:
    left, right = data.cut(500)
    assert len(left) == 500
    assert len(right) == 500

    with pytest.raises(ValueError):
        data.cut(1500)

    with pytest.raises(ValueError):
        data.cut(-500)


def test_data_float_split(data: Data) -> None:
    train, test = data.split(0.8)
    assert len(train) == 800
    assert len(test) == 200

    with pytest.raises(ValueError):
        data.split(1.5)

    with pytest.raises(ValueError):
        data.split(-0.5)


def test_data_int_split(data: Data) -> None:
    train, test = data.split(800)
    assert len(train) == 800
    assert len(test) == 200

    with pytest.raises(ValueError):
        data.split(1500)

    with pytest.raises(ValueError):
        data.split(-500)


def test_data_float_take(data: Data) -> None:
    sub_data = data.take(0.8)
    assert len(sub_data) == 800

    with pytest.raises(ValueError):
        data.take(1.5)

    with pytest.raises(ValueError):
        data.take(-0.5)


def test_data_int_take(data: Data) -> None:
    sub_data = data.take(800)
    assert len(sub_data) == 800

    with pytest.raises(ValueError):
        data.take(1500)

    with pytest.raises(ValueError):
        data.take(-500)


def test_data_featurize(data: Data) -> None:
    featurizer = SpatialFeaturizer()
    featurized_data = data.featurize(featurizer)
    assert len(featurized_data) == 1000
    assert len(featurized_data[0]) == featurizer.count


def test_map_transforms_pairs(data: Data) -> None:
    mapped = data.map(lambda traj, label: (traj, label.upper()))
    assert set(mapped.classes) == {"A", "B", "C"}


def test_filter_keeps_matching_pairs(data: Data) -> None:
    filtered = data.filter(lambda _, label: label == "a")
    assert set(filtered.classes) == {"a"}


def test_featurize_is_cached_per_featurizer(data: Data) -> None:
    featurizer = SpatialFeaturizer()
    assert data.featurize(featurizer) is data.featurize(featurizer)


def test_datasets_read_their_files(tmp_path: Path) -> None:
    traj = yupi.Trajectory(x=[0.0, 1.0], y=[0.0, 1.0], traj_id="0")
    path = tmp_path / "demo.json"
    content = {"version": 2, "trajs": [JSONSerializer.to_json(traj)], "labels": ["a"]}
    path.write_text(json.dumps(content), encoding="utf-8")
    loaded = Dataset.from_file(str(path), "demo")
    assert (loaded.name, loaded.version, len(loaded), loaded.labels) == ("demo", 2, 1, ["a"])


@pytest.mark.parametrize("missing", ["trajs", "labels"])
def test_incomplete_files_are_rejected(missing: str) -> None:
    content: dict = {"trajs": [], "labels": []}
    del content[missing]
    with pytest.raises(ValueError, match=f"{missing} not found"):
        Dataset._from_json("demo", content)


class _Git:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ls_remote(self, url: str, sort: str, tags: bool) -> str:
        self.calls.append(sort)
        return "abc\trefs/tags/v2\nabc\trefs/tags/v1"


def _serve_tags(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    def head(url: str, timeout: int) -> SimpleNamespace:
        found = any(f"/{tag}/" in url for tag in available)
        return SimpleNamespace(status_code=302 if found else 404)

    monkeypatch.setattr(dataset_module, "Git", _Git)
    monkeypatch.setattr(requests, "head", head)


def test_the_newest_release_with_the_file_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_tags(monkeypatch, {"v1"})
    assert Dataset._get_dataset_url("demo").endswith("/releases/download/v1/demo.zip")


def test_a_dataset_in_no_release_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_tags(monkeypatch, set())
    with pytest.raises(ValueError, match="demo"):
        Dataset._get_dataset_url("demo")


def _cached(name: str, content: str) -> Path:
    folder = Path(config.DS_DIR.format(name))
    folder.mkdir(parents=True)
    path = folder / f"{name}.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_a_cached_dataset_is_not_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    traj = yupi.Trajectory(x=[0.0, 1.0], y=[0.0, 1.0])
    _cached("demo", json.dumps({"trajs": [JSONSerializer.to_json(traj)], "labels": ["a"]}))
    monkeypatch.setattr(Dataset, "_get_dataset_url", staticmethod(lambda name: "unused"))
    monkeypatch.setattr(dataset_module, "download_dataset", lambda *a: pytest.fail("downloaded"))
    assert len(Dataset.get("demo")) == 1


def test_an_invalid_cache_is_downloaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    path = _cached("demo", "{}")
    traj = yupi.Trajectory(x=[0.0, 1.0], y=[0.0, 1.0])
    good = json.dumps({"trajs": [JSONSerializer.to_json(traj)], "labels": ["a"]})
    monkeypatch.setattr(Dataset, "_get_dataset_url", staticmethod(lambda name: "url"))
    monkeypatch.setattr(dataset_module, "download_dataset", lambda *a: path.write_text(good))
    assert len(Dataset.get("demo")) == 1


@pytest.mark.parametrize(
    "name",
    [
        "geolife",
        "animals",
        "mnist_stroke",
        "hurdat2",
        "cma_bst",
        "uci_gotrack",
        "uci_movement_libras",
        "uci_pen_digits",
        "uci_characters",
        "traffic",
        "diffusive_particles",
    ],
)
def test_named_loaders(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake(dataset_name: str, force: bool = False) -> str:
        calls.append((dataset_name, force))
        return "loaded"

    monkeypatch.setattr(Dataset, "_from_url", staticmethod(fake))
    assert getattr(Dataset, name)(redownload=True) == "loaded"
    assert calls == [(name, True)]
