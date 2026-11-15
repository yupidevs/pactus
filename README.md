<div align="center">

<img src="assets/logo.svg" alt="" width="96">

# pactus

**Framework to evaluate Trajectory Classification Algorithms**

[![QA](https://github.com/yupidevs/pactus/actions/workflows/qa.yml/badge.svg)](https://github.com/yupidevs/pactus/actions/workflows/qa.yml)
[![Release](https://img.shields.io/github/v/release/yupidevs/pactus?label=release)](https://github.com/yupidevs/pactus/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.05738/status.svg)](https://doi.org/10.21105/joss.05738)
[![Docs](https://readthedocs.org/projects/pactus/badge/?version=latest)](https://pactus.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/pactus)](https://pypi.org/project/pactus/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

<!-- Badges to move into the rows above once they apply:
[![Docker image](https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/yupidevs/pactus/pkgs/container/pactus)
-->

[Installation](#installation) · [Getting started](#getting-started) · [Available datasets](#available-datasets) · [Citing pactus](#citing-pactus) · [Development](#development)

</div>

---

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

pactus requires Python 3.11 or newer, and up to 3.13 for now, because TensorFlow has no wheels for newer versions yet.

You can install pactus from PyPI using pip:

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

It should produce an output as the following:

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

## Citing pactus

If you use pactus in your research, please cite [our article in the Journal of Open Source Software](https://doi.org/10.21105/joss.05738):

```bibtex
@article{pactus2023,
  title   = {pactus: A Python framework for trajectory classification},
  author  = {Viera-L{\'o}pez, G. and Morgado-Vega, J. J. and Reyes, A. and Altshuler, E. and Almeida-Cruz, Yudivi{\'a}n and Manganini, Giorgio},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {89},
  pages   = {5738},
  year    = {2023},
  doi     = {10.21105/joss.05738}
}
```

## Development

Setup, QA and releases: [DEVELOPMENT.md](DEVELOPMENT.md). See also the [pactus documentation](https://pactus.readthedocs.io/en/latest/).

## Contributing

Issues and pull requests are welcome; `make qa` must pass.

## License

MIT: see [LICENSE](LICENSE).

---

<sub>🚀 Project starter provided by [Cookie Pyrate v1.1.2](https://github.com/gvieralopez/cookie-pyrate/tree/v1.1.2)</sub>
