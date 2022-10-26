# pactus

Standing from *Path Classification Tools for Unifying Strategies*, **pactus**
is a Python library that allows testing different classification methods on
several trajectory datasets.

It comes with some built-in models and datasets according to the
state-of-the-art in trajectory classification. However, it is implemented in an
extensible way, so the users can build their own models and datasets.

> NOTE: Built-in datasets don't contain the raw trajectoy data. When a
> dataset is loaded for the first time it downloads the necessary data
> automatically.

## Instalation

```bash
pip install pactus
```

## Quick example

```python
from pactus import Featurizer, LangevinDataset, RandomForestModel, features

SEED = 0  # Random seed for reproducibility

# Load a dataset
dataset = LangevinDataset()

# Select the desired features to be extracted from the trajectories
featurizer = Featurizer(selected=features.ALL)

# Defining a model
model = RandomForestModel(
    featurizer=featurizer,
    bootstrap=False,
    random_state=SEED,
    n_jobs=6,
)

# Spliting dataset
train, test = dataset.split(0.8, random_state=SEED)

# Train the model
model.train(data=train, cross_validation=5)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Print the evaluation
evaluation.show()
```

**Output:**

```
Confusion matrix:

type_1      type_2      recall
====================================
99.5        0.0         100.0
0.5         100.0       99.5
------------------------------------
99.5        100.0
```
