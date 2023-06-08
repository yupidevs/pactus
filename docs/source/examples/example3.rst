Example 3
=========

In this example we illustrate how to evaluate several models available in pactus
in a single dataset.

The example is structured as follows:
  | :ref:`Setup dependencies 3`
  | :ref:`Definition of parameters 3`
  | :ref:`Loading Data 3`
  | :ref:`Loading the model 3`
  | :ref:`Training and evaluation 3`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/pactus/blob/master/examples/example_03.py>`_.

.. _Setup dependencies 3:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   from tensorflow import keras

   from pactus import Dataset, featurizers
   from pactus.models import (
       DecisionTreeModel,
       KNeighborsModel,
       LSTMModel,
       RandomForestModel,
       SVMModel,
       TransformerModel,
       XGBoostModel,
   )

.. _Definition of parameters 3:

2. Definition of parameters
---------------------------

We define a random seed for reproducibility

.. code-block:: python

   SEED = 0

.. _Loading Data 3:

3. Loading Data
---------------

To load the UCI Characters dataset we can simply do:

.. code-block:: python

   dataset = Dataset.uci_characters()

Then, we can create a train/test split of 80/20 respectively:

.. code-block:: python

   train, test = dataset.split(.8, random_state=SEED)

.. _Loading the model 3:

4. Loading the models
---------------------

Since we are going to use several models that are not able to deal with 
data of arbitrary length, we need to create an object
that converts every trajectory into a fixed size feature vector. In this case,
we are going to use the UniversalFeaturizer for all those models. This featurizer
includes all available features.

.. code-block:: python
   
   featurizer = featurizers.UniversalFeaturizer()

We can start by creating all the models requiring the featurizer and storing them
in a list:

.. code-block:: python

   vectorized_models = [
       RandomForestModel(
           featurizer=featurizer,
           max_features=16,
           n_estimators=200,
           bootstrap=False,
           warm_start=True,
           n_jobs=6,
           random_state=SEED,
       ),
       KNeighborsModel(
           featurizer=featurizer,
           n_neighbors=7,
       ),
       DecisionTreeModel(
           featurizer=featurizer,
           max_depth=7,
           random_state=SEED,
       ),
       SVMModel(
           featurizer=featurizer,
           random_state=SEED,
       ),
       XGBoostModel(
           featurizer=featurizer,
           random_state=SEED,
       ),
   ]

Then, we proceed to create the LSTM and Transformer models without the featurizer
since both of them can handle trajectories directly:

.. code-block:: python
   
   lstm = LSTMModel(
       loss="sparse_categorical_crossentropy",
       optimizer="rmsprop",
       metrics=["accuracy"],
       random_state=SEED,
   )

   model = TransformerModel(
       head_size=512,
       num_heads=4,
       num_transformer_blocks=4,
       optimizer=keras.optimizers.Adam(learning_rate=1e-4),
       random_state=SEED,
   )

.. _Training and evaluation 3:

5. Training and evaluation
--------------------------

Training and evaluation of the models requiring the featurizer can be achieved by:

.. code-block:: python

   for model in vectorized_models:
       print(f"\nModel: {model.name}\n")
       model.train(train, cross_validation=5)
       evaluation = model.evaluate(test)
       evaluation.show()

LSTM training and evaluation can be conducted by:

.. code-block:: python

   checkpoint = keras.callbacks.ModelCheckpoint(
       f"partially_trained_model_lstm_{dataset.name}.h5",
       monitor="loss",
       verbose=1,
       save_best_only=True,
       mode="min",
   )
   lstm.train(train, dataset, epochs=20, checkpoint=checkpoint)
   evaluation = lstm.evaluate(test)
   evaluation.show()

Similarly, Transformer evaluation can be performed by:

.. code-block:: python

   checkpoint = keras.callbacks.ModelCheckpoint(
       f"partially_trained_model_transformer_{dataset.name}.h5",
       monitor="loss",
       verbose=1,
       save_best_only=True,
       mode="min",
   )
   transformer.train(train, dataset, epochs=150, checkpoint=checkpoint)
   evaluation = transformer.evaluate(test)
   evaluation.show()

Each model should output the performance results using different metrics and they
can be fairly compared among each other since the data used for training and evaluation 
was identical.

