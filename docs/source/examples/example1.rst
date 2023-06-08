Example 1
=========

We illustrate how to evaluate a Random Forest Classifier on a preprocessed
version of the GeoLife Dataset. This examples seeks to reproduce the preprocessing
conducted in [1]

The example is structured as follows:
  | :ref:`Setup dependencies 1`
  | :ref:`Definition of parameters 1`
  | :ref:`Loading Data 1`
  | :ref:`Loading the model 1`
  | :ref:`Training and evaluation 1`
  | :ref:`References 1`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/pactus/blob/master/examples/example_01.py>`_.

.. _Setup dependencies 1:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   from pactus import Dataset, featurizers
   from pactus.models import RandomForestModel


.. _Definition of parameters 1:

2. Definition of parameters
---------------------------

We define a random seed for reproducibility

.. code-block:: python

   SEED = 0

.. _Loading Data 1:

3. Loading Data
---------------

To load the original GeoLife dataset we can simply do:

.. code-block:: python

   dataset = Dataset.geolife()

Then, we can process it to keep only the desired classes, combine similar classes
and create a train/test split as proposed on [1]:

.. code-block:: python

   # Classes that are going to be used
   use_classes = {"car", "taxi-bus", "walk", "bike", "subway", "train"}

   # Preprocess the dataset and split it into train and test sets
   train, test = (
       # Remove short and poorly time sampled trajectories
       dataset.filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)
       # Join "taxi" and "bus" into "taxi-bus"
       .map(lambda _, label: (_, "taxi-bus" if label in ("bus", "taxi") else label))
       # Only use the classes defined in use_classes
       .filter(lambda _, label: label in use_classes)
       # Split the dataset into train and test
       .split(train_size=0.7, random_state=SEED)
   )


.. _Loading the model 1:

4. Loading the model
--------------------

Since we are going to use a Random Forest model, we need to create an object
that converts every trajectory into a fixed size feature vector. In this case,
we are going to use the UniversalFeaturizer, which includes all available features
on pactus:

.. code-block:: python

   featurizer = featurizers.UniversalFeaturizer()

Then, we can create the desired model using the aforementioned featurizer:

.. code-block:: python

   model = RandomForestModel(
       featurizer=featurizer,
       max_features=16,
       n_estimators=200,
       bootstrap=False,
       random_state=SEED,
       warm_start=True,
       n_jobs=6,
   )

.. _Training and evaluation 1:

5. Training and evaluation
--------------------------

Training and evaluation can be conducted as follows:

.. code-block:: python

   # Train the model
   model.train(data=train, cross_validation=5)

   # Evaluate the model on a test dataset
   evaluation = model.evaluate(test)

   # Show the evaluation results
   evaluation.show()

Evaluation results should look like:

.. code-block:: text

   General statistics:
   Accuracy: 0.913
   F1-score: 0.892
   Mean precision: 0.910
   Mean recall: 0.877
   
   Confusion matrix:
   bike      car       subway    taxi-bus  train     walk      precision 
   ======================================================================
   89.83     0.56      0.74      1.29      0.0       1.41      94.03     
   0.25      79.1      0.74      1.94      0.0       0.11      90.32     
   0.0       0.56      82.35     1.46      0.0       0.22      90.32     
   2.48      18.08     10.29     91.42     12.12     2.5       87.19     
   0.0       0.56      0.0       0.32      87.88     0.0       90.62     
   7.44      1.13      5.88      3.56      0.0       95.76     93.43     
   ----------------------------------------------------------------------
   89.83     79.1      82.35     91.42     87.88     95.76     


.. _References 1:

6. References
-------------
| [1] Zheng, Yu, et al. "Understanding mobility based on GPS data." Proceedings of the 10th international conference on Ubiquitous computing. 2008.
