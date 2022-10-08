from random import Random
from typing import Any

from yupi import Trajectory
from yupi.generators import LangevinGenerator

from yuca.dataset import Dataset

# Dataset metadata
NAME = "langevin"
VERSION = "0.1.0"  # See version description in config.py


class LangevinDataset(Dataset):
    """Dataset of trajectories generated from the Langevin model."""

    def __init__(self, reyupify: bool = False):
        super().__init__(NAME, VERSION, False, reyupify)

    def yupify(self) -> tuple[list[Trajectory], list[Any]]:
        trajs, labels = [], []
        seed = 0

        def get_seed() -> int:
            nonlocal seed
            seed += 1
            return seed

        rng = Random(0)

        total_time_mean = 10
        dt_mean = 0.1
        labels_config = {
            "type_1": {
                "gamma": 0.1,
                "sigma": 0.1,
                "count": 100,
            },
            "type_2": {
                "gamma": 0.1001,
                "sigma": 0.1,
                "count": 100,
            },
        }

        for label, config in labels_config.items():
            assert isinstance(config["count"], int) and config["count"] > 0
            for _ in range(config["count"]):
                gamma, sigma = config["gamma"], config["sigma"]
                lg_gen = LangevinGenerator(
                    dim=2,
                    N=1,
                    dt=abs(rng.normalvariate(dt_mean, 0.1)),
                    T=abs(rng.normalvariate(total_time_mean, 2)),
                    seed=get_seed(),
                    gamma=gamma,
                    sigma=sigma,
                )
                trajs.append(lg_gen.generate()[0])
                labels.append(label)

        return trajs, labels
