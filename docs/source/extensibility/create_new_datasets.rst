Create new datasets
===================

To craft a new dataset you only need to instanciate a new `Dataset` providing a
name, the trajectories (yupi trajectory instances) and the labels (e.g., ``str``
or ``int``).

Here is an example of a simple dataset of 1D trajectories being created from scratch:

.. code-block:: python
    
    from yupi import Trajectory 
    from pactus import Dataset
    
    trajs = [
        Trajectory(x=[9,7,4,3,2]),
        Trajectory(x=[1,2,3,4]),
        Trajectory(x=[4,7,8,10,11]),
        Trajectory(x=[3,2,0,-1]),
    ]
    labels = [
        'backward',
        'forward',
        'forward',
        'backward',
    ]
    dummy_ds = Dataset("dummy", trajs, labels)

.. note::
    If you want to share a trajectory dataset with the community, consider
    following this guide: :ref:`adding-datasets`.
    