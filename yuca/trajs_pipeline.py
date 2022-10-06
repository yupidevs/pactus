from random import Random
from typing import Any

import numpy as np
from yupi.generators import DiffDiffGenerator, LangevinGenerator
from yupi.trajectory import Trajectory

from yuca.data_handler import (
    get_selected_data,
    load_trajs_data,
    load_trajs_metadata,
    traj_class_builder,
)
from yuca.data_parser import parse_data
from yuca.pipeline import Pipeline, PipelineStep

GEOLIFE_DATASET = 0
LANGEVIN_GENERATED = 1
DIFF_DIFF_GENERATED = 2
LANGEVIN_VS_DIFF_DIFF = 3


def _generate_langevine(N, dt, tt, gamma, sigma, seed) -> list[Trajectory]:
    lg = LangevinGenerator(dim=2, N=N, dt=dt, T=tt, seed=seed, gamma=gamma, sigma=sigma)
    return lg.generate()


def _generate_diffdiff(N, dt, tt, gamma, sigma, seed) -> list[Trajectory]:
    lg = DiffDiffGenerator(
        dim=2, N=N, dt=dt, T=tt, seed=seed, tau=1 / gamma, sigma=sigma
    )
    return lg.generate()


@PipelineStep.build("langevin dataset generator")
def _get_langevin_dataset(total=50) -> tuple[list[Trajectory], np.ndarray]:
    if isinstance(total, tuple):
        total = 50
    dataset = []
    classes = []

    seed = 0

    def get_seed() -> int:
        nonlocal seed
        seed += 1
        return seed

    rng = Random(0)
    lengths = [10 for _ in range(total)]
    types = [[0.1, 0.1], [0.1001, 0.1]]

    for j, k in enumerate(types):
        temp = []
        for i in lengths:
            gamma, sigma = k
            temp += _generate_langevine(
                1,
                abs(rng.normalvariate(0.1, 0.1)),
                abs(rng.normalvariate(i, 2)),
                gamma,
                sigma,
                get_seed(),
            )
            dataset += temp
            classes += [f"type{j}"] * len(temp)

    return dataset, np.array(classes)


@PipelineStep.build("diff diff dataset generator")
def _get_diffdiff_dataset(total=50) -> tuple[list[Trajectory], np.ndarray]:
    if isinstance(total, tuple):
        total = 50
    dataset = []
    classes = []

    seed = 0

    def get_seed() -> int:
        nonlocal seed
        seed += 1
        return seed

    rng = Random(0)
    lengths = [10 for _ in range(total)]
    types = [[0.1, 0.1], [0.1001, 0.1]]

    for j, k in enumerate(types):
        temp = []
        for i in lengths:
            gamma, sigma = k
            temp += _generate_diffdiff(
                1,
                abs(rng.normalvariate(0.1, 0.1)),
                abs(rng.normalvariate(i, 2)),
                gamma,
                sigma,
                get_seed(),
            )
            dataset += temp
            classes += [f"type{j}"] * len(temp)

    return dataset, np.array(classes)


@PipelineStep.build("langevin vs diff diff dataset generator")
def _get_lan_vs_diff_dataset(_none) -> tuple[list[Trajectory], np.ndarray]:
    count = 5
    lang = _get_langevin_dataset(count)[0]
    diff = _get_diffdiff_dataset(count)[0]
    dataset = lang + diff
    classes = ["langevin"] * len(lang) + ["diff-diff"] * len(diff)
    return dataset, np.array(classes)


def get_traj_extraction_pl(origin: int = GEOLIFE_DATASET) -> Pipeline:
    if origin == GEOLIFE_DATASET:
        return Pipeline(
            "Geolife data loader",
            parse_data,
            load_trajs_metadata,
            get_selected_data,
            load_trajs_data,
            traj_class_builder,
        )
    if origin == LANGEVIN_GENERATED:
        return Pipeline(
            "Langevin generated",
            _get_langevin_dataset,
        )
    if origin == DIFF_DIFF_GENERATED:
        return Pipeline(
            "Diff diff generated",
            _get_diffdiff_dataset,
        )
    if origin == LANGEVIN_VS_DIFF_DIFF:
        return Pipeline(
            "Langevin vs diff diff generated",
            _get_lan_vs_diff_dataset,
        )
    raise ValueError("Invalid origin")
