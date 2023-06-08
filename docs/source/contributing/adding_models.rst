.. _adding-models:

Adding models to pactus
=======================

1. Create the model script
--------------------------

You should start by writting your model as a pactus model class. 
Refer to :ref:`Create new models` section if you need help with that.

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
