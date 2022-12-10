from tensorflow import keras

from pactus import Dataset, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    RandomForestModel,
    SVMModel,
    TransformerModel,
)

SEED = 0  # Random seed for reproducibility

featurizer = featurizers.UniversalFeaturizer()
models = [
    RandomForestModel(
        featurizer=featurizer,
        max_features=16,
        n_estimators=200,
        bootstrap=False,
        random_state=SEED,
        warm_start=True,
        n_jobs=6,
    ),
    KNeighborsModel(
        featurizer=featurizer,
        n_neighbors=7,
    ),
    DecisionTreeModel(
        featurizer=featurizer,
        max_depth=7,
    ),
    SVMModel(
        featurizer=featurizer,
    ),
    TransformerModel(
        head_size=256,
        num_heads=1,
        num_transformer_blocks=2,
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    ),
]

dataset = Dataset.mnist_stroke()
train, test = dataset.take(10_000).split(train_size=0.7, random_state=SEED)

for model in models:
    print(f"\nModel: {dataset.name}\n")

    model.train(train, cross_validation=5)
    evaluation = model.evaluate(test)
    evaluation.show()
