from collections import Counter
from typing import Any

import numpy as np
from yupi import Trajectory

import yuca.features as feats
from yuca.pipeline import Pipeline, PipelineStep
from yuca.extra_pl_steps import dataset_description


def get_feat_extraction_pl(features: int = feats.ALL_FEATS, **kwargs) -> Pipeline:
    """Returns a PipelineStep that extracts features from a trajectory.

    Parameters
    ----------
    features : int
        Features to extract.

    Returns
    -------
    PipelineStep
        PipelineStep that extracts features from a trajectory.
    """


    @PipelineStep.build("feat_extractor")
    def feat_extractor_step(
        trajs: list[Trajectory], classes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from a trajectory.

        Input: list of trajectoies, classes
        Output: list of feature vectors, classes
        """

        def _get_feat(i) -> np.ndarray:
            print(f"{(i+1)/len(trajs):.2%}", end="\r")
            return feats.get_feat_vector(trajs[i], features, **kwargs)

        return np.stack([_get_feat(i) for i in range(len(trajs))]), classes

    return Pipeline("Feature extractor", dataset_description, feat_extractor_step)
