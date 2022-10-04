from collections import Counter

import numpy as np
from yupi import Trajectory

from pipeline import PipelineStep


@PipelineStep.build("dataset description")
def dataset_description(
    trajs: list[Trajectory], classes: np.ndarray
) -> tuple[list[Trajectory], np.ndarray]:
    """
    Prints a description of the dataset

    Input: list of trajectoies, classes
    Output: list of feature vectors, classes
    """
    print(f"Total trajectories: {len(trajs)}")
    counter = Counter(classes)
    for cls_name, cls_count in counter.most_common(len(counter)):
        print(f"{cls_name:>15}: {cls_count:<10}")
    return trajs, classes
