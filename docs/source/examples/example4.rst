Example 4
=========

In this example we illustrate how to evaluate a single model in all available 
datasets.

The example is structured as follows:
  | :ref:`Setup dependencies 4`
  | :ref:`Definition of parameters 4`
  | :ref:`Loading Data 4`
  | :ref:`Loading the model 4`
  | :ref:`Training and evaluation 4`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/pactus/blob/master/examples/example_04.py>`_.

.. _Setup dependencies 4:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   from pactus import Dataset, featurizers
   from pactus.models import XGBoostModel

.. _Definition of parameters 4:

2. Definition of parameters
---------------------------

We define a random seed for reproducibility

.. code-block:: python

   SEED = 0

.. _Loading Data 4:

3. Loading Data
---------------

We start by loading all the datasets available and storing them in a list:

.. code-block:: python

   datasets = [
       Dataset.geolife(),
       Dataset.animals(),
       Dataset.hurdat2(),
       Dataset.cma_bst(),
       Dataset.mnist_stroke(),
       Dataset.uci_pen_digits(),
       Dataset.uci_gotrack(),
       Dataset.uci_characters(),
       Dataset.uci_movement_libras(),
   ]


.. _Loading the model 4:

4. Loading the model
--------------------

Since we are going to use XGBoost model, and it is not able to deal with 
data of arbitrary length, we need to create an object
that converts every trajectory into a fixed size feature vector. In this case,
we are going to use the UniversalFeaturizer for all those models. This featurizer
includes all available features.

.. code-block:: python
   
   featurizer = featurizers.UniversalFeaturizer()

Then, we will need to create a model for each dataset and train them independently.

.. _Training and evaluation 4:

5. Training and evaluation
--------------------------

We iterate over all the available datasets and train an XGBoost model for each of them.
Notice that for better perfomance, we should filter out short trajectories and
classes with a small count of trajectories.

.. code-block:: python

   for dataset in datasets:
       print(f"\nDataset: {dataset.name}\n")

       # Split the dataset into train and test and filter out short trajectories
       train, test = dataset.filter(
           lambda traj, label: len(traj) >= 5
           and traj.r.delta.norm.sum() > 0
           and dataset.label_counts[label] > 5
       ).split(
           train_size=0.7,
           random_state=SEED,
       )

       # Define the model
       model = XGBoostModel(featurizer=featurizer)

       # Evaluate the results
       model.train(data=train, cross_validation=5)
       evaluation = model.evaluate(test)
       evaluation.show()


Each model should output the performance results using different metrics and they
can be fairly compared among each other since the data used for training and evaluation 
was identical.

