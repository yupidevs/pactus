import random
from typing import Any, Tuple

from pactus.dataset import Data, Dataset


def _get_zs(
    traj_len: int,
    size: int,
    zoom: Any = -1,
    shift: Any = -1,
) -> Tuple[int, int]:
    assert traj_len > size, f"{traj_len} <= {size}"
    if isinstance(zoom, int):
        _zoom = random.randint(1, traj_len // size) if zoom == -1 else zoom
    elif isinstance(zoom, list):
        _zoom = random.choice(zoom)
    elif isinstance(zoom, tuple):
        _zoom = random.randint(zoom[0], zoom[1])
    else:
        _zoom = zoom()

    if isinstance(shift, int):
        _shift = random.randint(0, traj_len - _zoom * size) if shift == -1 else shift
    elif isinstance(shift, list):
        _shift = random.choice(shift)
    elif isinstance(shift, tuple):
        _shift = random.randint(shift[0], shift[1])
    else:
        _shift = shift()

    assert isinstance(_zoom, int), f"{_zoom} is not an int"
    assert isinstance(_shift, int), f"{_shift} is not an int"
    assert (
        _shift + _zoom * size <= traj_len
    ), f"{_shift} + {_zoom} * {size} > {traj_len}"
    return _zoom, _shift


def zoomletize(data: Data, zoom: int = -1, shift: int = -1, size: int = -1) -> Data:
    new_trajs, new_labels = [], []
    size = 200 if size == -1 else size
    for traj, label in zip(data.trajs, data.labels):
        if len(traj) > size:
            _zoom, _shift = _get_zs(len(traj), size, zoom, shift)
            new_trajs.append(traj[_shift : _shift + _zoom * size : _zoom])
            assert len(new_trajs[-1]) == size, f"{len(new_trajs[-1])} != {size}"
        else:
            new_trajs.append(traj)
        new_labels.append(label)
    return Dataset("zoomletized_data", new_trajs, new_labels, 0)
