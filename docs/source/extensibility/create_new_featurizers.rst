Create new featurizers
======================

To craft a new featurizer you only need to create a derived class from the
``yupi.core.featurizers.Featurizer``. Here is an example of a dummy featurizer
that uses the initial and final position on the x axis as the only two features.
You can use it for guidance when implementing your own methods to compute
feature vectors from trajectories:

.. code-block:: python

    # Import necessary libraries
    import numpy as np
    from yupi.core.featurizers import Featurizer
    from yupi import Trajectory
    from typing import List

    class DummyFeaturizer(Featurizer):
        def __init__():
            super().__init__()
        
        def featurize(self, trajs: List[Trajectory]) -> np.ndarray:
            feature_vectors = []
            for traj in trajs:
                features = [traj.r.x[0], traj.r.x[-1]]
                feature_vectors.append(features)
            return np.array(feature_vectors)

        @property
        def count(self) -> int:
            # This must return the total number of features.
            # This is for optimizing the feature computation.
            return 2

.. note::
    This dummy featurizer is absurd for practical purposes. We are only
    showing it to illustrate how you can create your own. Any method you
    implement for converting trajectories into vectors should capture important
    statistical patterns from your trajectories.