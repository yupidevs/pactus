.. _adding-datasets:

Adding datasets to pactus
=========================

1. Add the dataset to the trajectory-datasets repository
--------------------------------------------------------

For this follow the instructions on `Adding datasets to this
repository <https://github.com/yupidevs/trajectory-datasets#adding-datasets-to-this-repository>`_.

2. Use the dataset
------------------

Once the dataset is merged and published in the `trajectory-datasets
<https://github.com/yupidevs/trajectory-datasets>`_ repository, you can use it
from pactus by calling the ``Dataset.get`` method like this

.. code-block:: python

    from pactus import Dataset

    your_ds = Dataset.get("your_ds_name")