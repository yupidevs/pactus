.. _Create new models:

Create new models
=================

To craft a new model you only need to create a derived class from ``Model``.
Here is a template you can use for guidance:

.. code-block:: python

  # Import necessary libraries
  from typing import Any, List
  from yupi import Trajectory
  from pactus import featurizers
  from pactus.dataset import Data
  from pactus.models.model import Model

  NAME = "example" # Write here your model name

  # Change the class name according to the model name.
  class ExampleModel(Model):
      def __init__(self):
          # Base class only needs the model name
          super().__init__(NAME)

          # Define here new model fields if needed
          # ...

      def train(self, data: Data, cross_validation: int = 0):
          # Implement here the train method of the model

      def predict(self, data: Data) -> List[Any]:
          # This method should give the (pre-trained) model predictions for
          # the trajectories in `data`.


For example, here is the pactus model implementation of the KNN model:

.. code-block:: python

  from typing import Any, List

  from sklearn.model_selection import GridSearchCV
  from sklearn.neighbors import KNeighborsClassifier
  from yupi import Trajectory

  from pactus import featurizers
  from pactus.dataset import Data
  from pactus.models.model import Model

  NAME = "kneighbors"


  class KNeighborsModel(Model):
      """Implementation of a K-Nearst Neighbors Classifier."""

      def __init__(self, featurizer: featurizers.Featurizer, **kwargs):
          super().__init__(NAME)
          self.featurizer = featurizer
          self.model = KNeighborsClassifier(**kwargs)
          self.grid: GridSearchCV

          # This stores the configuration in the model summary.
          # This summary is only used as the model metadata.
          self.set_summary(**kwargs)

      def train(self, data: Data, cross_validation: int = 0):
          self.set_summary(cross_validation=cross_validation)
          x_data = data.featurize(self.featurizer)
          self.grid = GridSearchCV(self.model, {}, cv=cross_validation, verbose=3)
          self.grid.fit(x_data, data.labels)

      def predict(self, data: Data) -> List[Any]:
          x_data = data.featurize(self.featurizer)
          return self.grid.predict(x_data)

      def predict_single(self, traj: Trajectory) -> Any:
          """Predicts the label of a single trajectory."""
          return self.grid.predict([traj])[0]

.. note::
    If you want to share a model with the community, consider following this
    guide: :ref:`adding-models`.