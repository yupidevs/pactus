.. pactus documentation master file, created by
   sphinx-quickstart on Wed Jun  7 00:28:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pactus' documentation!
=================================

Standing from Path Classification Tools for Unifying Strategies, `pactus`
is a Python library that allows testing different classification methods on
several trajectory datasets.

It comes with some built-in models and datasets according to the
state-of-the-art in trajectory classification. However, it is implemented in an
extensible way, so the users can build their own models and datasets.

.. figure:: /images/example.jpg
   :alt: Minimal example of a pactus classification task
   :align: center
   :width: 550
   
   *Overview of the resources available in pactus coupled with an usage example*

.. note::
   Code from the example shown above:

   .. code-block:: python

      from pactus import Dataset, featurizers
      from pactus.models import RandomForestModel

      dataset = Dataset.animals()
      train, test = dataset.split(0.9)

      ft = featurizers.UniversalFeaturizer()
      model = RandomForestModel(featurizer=ft)
      model.train(train, cross_validation=5)

      evaluation = model.evaluate(test)
      evaluation.show()


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
  
   getting_started/installation
   getting_started/support

.. toctree::
   :maxdepth: 2
   :caption: Examples
  
   examples/example0
   examples/example1
   examples/example2
   examples/example3
   examples/example4

.. toctree::
   :maxdepth: 2
   :caption: Extensibility
  
   extensibility/create_new_datasets
   extensibility/create_new_models
   extensibility/create_new_featurizers

.. toctree::
   :maxdepth: 2
   :caption: Contributing
  
   contributing/engaging
   contributing/adding_datasets
   contributing/adding_models

.. toctree::
   :maxdepth: 2
   :caption: Advanced Resources

   available_datasets/available_datasets
   api_reference/api_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
