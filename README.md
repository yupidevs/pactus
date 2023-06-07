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

This is quick example of how to test a Random Forest classifier on a preprocessed
version of the GeoLife dataset:

```python
from pactus import Dataset, featurizers
from pactus.models import RandomForestModel

# Load Dataset
dataset = Dataset.geolife()

# Select the desired features to be extracted from the trajectories
featurizer = featurizers.UniversalFeaturizer()

# Defining the model
model = RandomForestModel(
    featurizer=featurizer,
    max_features=16,
    n_estimators=200,
    bootstrap=False,
    random_state=SEED,
    warm_start=True,
    n_jobs=6,
)

# Preprocess the dataset and split it into train and test sets
use_classes = {"car", "taxi-bus", "walk", "bike", "subway", "train"}
train, test = (
    # Remove short and pourly time sampled trajectories
    dataset.filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)
    # Join "taxi" and "bus" into "taxi-bus"
    .map(lambda _, label: (_, "taxi-bus" if label in ("bus", "taxi") else label))
    # Only use the classes defined in use_classes
    .filter(lambda _, label: label in use_classes)
    # Split the dataset into train and test
    .split(train_size=0.7, random_state=SEED)
)

# Train the model
model.train(data=train, cross_validation=5)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Show the evaluation results
evaluation.show()
```

It should output the evaluation results as:

```
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
```

## Available datasets

See the whole [list of datasets](https://github.com/yupidevs/trajectory-datasets) compatible with pactus


## Contributing

Follow the guidlines from [pactus documentation](https://pactus.readthedocs.io/en/latest/)