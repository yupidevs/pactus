Example 2
=========

We illustrate how to evaluate a Transformer Network for classifying the trajectories
of the MNIST stroke dataset. This examples seeks to partially reproduce the results
reported in [1]

The example is structured as follows:
  | :ref:`Setup dependencies 2`
  | :ref:`Definition of parameters 2`
  | :ref:`Loading Data 2`
  | :ref:`Loading the model 2`
  | :ref:`Training and evaluation 2`
  | :ref:`References 2`

.. note::
   You can access `the script of this example <https://github.com/yupidevs/pactus/blob/master/examples/example_02.py>`_.

.. _Setup dependencies 2:

1. Setup dependencies
---------------------

Import all the dependencies:

.. code-block:: python

   from tensorflow import keras
   from pactus import Dataset
   from pactus.models import TransformerModel


.. _Definition of parameters 2:

2. Definition of parameters
---------------------------

We define a random seed for reproducibility

.. code-block:: python

   SEED = 0

.. _Loading Data 2:

3. Loading Data
---------------

To load the MNIST stroke dataset we can simply do:

.. code-block:: python

   dataset = Dataset.mnist_stroke()

Then, we can create a train/test split as proposed on [1]:

.. code-block:: python

   train, test = dataset.cut(60_000)

.. _Loading the model 2:

4. Loading the model
--------------------

Since transformers are able to deal with data of arbitrary length, there is no need
to create a featurizer for this model, and we can directly use it:

.. code-block:: python

   model = TransformerModel(
       head_size=512,
       num_heads=4,
       num_transformer_blocks=4,
       optimizer=keras.optimizers.Adam(learning_rate=1e-4),
   )

.. _Training and evaluation 2:

5. Training and evaluation
--------------------------

Training and evaluation can be conducted as follows:

.. code-block:: python

   # Train the model on the train dataset
   model.train(train, dataset, epochs=150, batch_size=64, checkpoint=checkpoint)

   # Evaluate the model on a test dataset
   evaluation = model.evaluate(test)

   # Print the evaluation
   evaluation.show()

Evaluation results should look like:

.. code-block:: text

   [Coming soon] 


.. _References 2:

6. References
-------------
| [1] BAE, Keywoong; LEE, Suan; LEE, Wookey. Transformer Networks for Trajectory Classification. En 2022 IEEE International Conference on Big Data and Smart Computing (BigComp). IEEE, 2022. p. 331-333.