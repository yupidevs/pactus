Adding models
=============

1. Create the model script
--------------------------

The first thing you need to do is to write your model as a pactus
model class. Here is a template you can use for guidance:

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

With ease, you can utilize your model on all available Pactus datasets in an
ad-hoc manner. However, if you are interested in adding the new model to the
pactus library, you can refer to steps 2 and 3 for guidance.

2. Fork the pactus GitHub repository and add the model script
-------------------------------------------------------------

.. note::
   You must have ``git`` installed and a GitHub account to fork and clone the
   project.

To fork the project open this link: `Fork pactus repository
<https://github.com/yupidevs/pactus/fork>`_ and click on the **Create fork**
button.

This will create a fork of project in your personal repository. The url
should be like this::

  https://github.com/<user-name>/pactus

Clone it into your local machine and enter into the project folder. This can be
done by running::

  $ git clone https://github.com/<user-name>/pactus
  $ cd pactus

Next, create a new branch::

  $ git checkout -b feat/add-model

Put the model script in the models folder (``./pactus/models/``) along with the
already implemented ones and include the necessary ``import`` in the ``__init__.py``
file on the same folder.

.. note::
  It is recommended to also add the modell class name in the ``__all__`` variable
  in the ``./pactus/models/__init__.py`` file.

Finally, push the branch to upstream::

  $ git push --set-upstream origin feat/add-model

3. Make a pull request
----------------------

The last step is to make a pull request. To do this, go to your pactus
local repository (the fork) pulls tab::

  https://github.com/<user-name>/pactus/pulls

and create a Pull Request from your local branch ``feat/add-model`` to the
original repository branch ``main``.

After that, we will analyze your changes and review the pull request
before the final merge.
