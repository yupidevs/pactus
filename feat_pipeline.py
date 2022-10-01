from yupi import Trajectory

import features as feats
from pipeline import Pipeline, PipelineStep


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
        trajs: list[Trajectory], classes: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """Extract features from a trajectory.

        Input: list of trajectoies, classes
        Output: list of feature vectors, classes
        """

        def _get_feat(i):
            print(f"{(i+1)/len(trajs):.2%}", end="\r")
            return feats.get_feat_vector(trajs[i], features, **kwargs)

        return [_get_feat(i) for i in range(len(trajs))], classes

    return Pipeline("Feature extractor", feat_extractor_step)
