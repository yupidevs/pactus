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

## Installation

Make sure you have a Python interpreter newer than version 3.8:

```bash
❯ python --version
Python 3.8.0
```

Then, you can simply install pactus from pypi using pip:

```bash
pip install pactus
```

## Getting started

This is quick example of how to test a Random Forest classifier on the Animals dataset:

```python
from pactus import Dataset, featurizers
from pactus.models import RandomForestModel

SEED = 0

# Load dataset
dataset = Dataset.animals()

# Split data into train and test subsets
train, test = dataset.split(0.9, random_state=SEED)

# Convert trajectories into feature vectors
ft = featurizers.UniversalFeaturizer()

# Build and train the model
model = RandomForestModel(featurizer=ft, random_state=SEED)
model.train(train, cross_validation=5)

# Evaluate the results on the test subset
evaluation = model.evaluate(test)
evaluation.show()
```

It should output evaluation results similar to:

```text
General statistics:

Accuracy: 0.885
F1-score: 0.849
Mean precision: 0.865
Mean recall: 0.850

Confusion matrix:

Cattle  Deer    Elk     precision
================================
75.0    0.0     0.0     100.0
25.0    80.0    0.0     66.67
0.0     20.0    100.0   92.86
--------------------------------
75.0    80.0    100.0
```

> ℹ️ Notice that by setting the random state to a fixed seed, we ensure the
reproducibility of the results. By changing the seed value, results may be
slightly different due to the stochastic processes used when splitting the
dataset and training the model.

## Available datasets

See the whole [list of datasets](https://github.com/yupidevs/trajectory-datasets) compatible with pactus

## Contributing

Follow the guidlines from [pactus documentation](https://pactus.readthedocs.io/en/latest/)
