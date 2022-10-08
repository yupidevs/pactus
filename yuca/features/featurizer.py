from pathlib import Path

import numpy as np
from yupi import Trajectory

from yuca import config
from yuca.dataset import Dataset, DatasetSlice
from yuca.dataset._utils import _get_path
from yuca.features.features import get_feat_vector

FeatureMask = int


class Featurizer:
    """
    The featurizer class is used to compute selected features of a given
    trajectory or dataset.
    """

    def __init__(self, selected: FeatureMask, recompute: bool = False, **kwargs):
        self.selected = selected
        self.recompute = recompute
        self.kwargs = kwargs

    def _recompute_required(self, feat_file: Path) -> bool:
        return self.recompute or not feat_file.exists()

    def _recompute(self, dataset: Dataset, feat_file: Path) -> np.ndarray:
        feats = np.stack(
            [
                get_feat_vector(traj, self.selected, **self.kwargs)
                for traj in dataset.trajs
            ]
        )
        np.savetxt(str(feat_file), feats)
        return feats

    def compute(self, data: Dataset | DatasetSlice) -> np.ndarray:
        """Computes the features matrix for a given dataset or slice."""

        feats = None
        dataset = data.dataset
        file_name = f"{self.selected}.txt"
        feat_file = _get_path(config.DS_FEATS_DIR, dataset.name) / file_name

        # recompute if required
        if self._recompute_required(feat_file):
            feats = self._recompute(dataset, feat_file)

        return feats or np.loadtxt(feat_file)

    def traj2vec(self, traj: Trajectory) -> np.ndarray:
        """Computes the features vector for a given trajectory."""
        return get_feat_vector(traj, self.selected, **self.kwargs)
