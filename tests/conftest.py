from pathlib import Path

import numpy as np
import pytest
from yupi import Trajectory

from pactus import Dataset, config


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DS_DIR", str(tmp_path / "cache" / "{0}"))
    monkeypatch.setattr(config, "DS_EVALS_DIR", str(tmp_path / "cache" / "{0}" / "evaluations"))


@pytest.fixture
def dataset() -> Dataset:
    rng = np.random.default_rng(0)
    trajs, labels = [], []
    for index in range(32):
        kind = index % 2
        x = np.cumsum(rng.normal(kind, 0.2, 20))
        y = np.cumsum(rng.normal(0, 0.2 + kind, 20))
        trajs.append(Trajectory(x=x, y=y, dt=1.0, traj_id=str(index)))
        labels.append("ab"[kind])
    return Dataset("synthetic", trajs, labels)
