from typing import List

from tensorflow import keras

from pactus import Dataset, Evaluation, EvaluationComparison, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    RandomForestModel,
    TransformerModel,
)

SEED = 0  # Random seed for reproducibility

evaluations: List[Evaluation] = []

# Load Dataset
dataset = Dataset.stochastic_models()

# Select the desired features to be extracted from the trajectories
featurizer = featurizers.UniversalFeaturizer()

# Defining the model
models = [
    KNeighborsModel(
        featurizer=featurizer,
        n_neighbors=7,
    ),
    DecisionTreeModel(
        featurizer=featurizer,
        max_depth=7,
    ),
    RandomForestModel(
        featurizer=featurizer,
        max_features=16,
        n_estimators=200,
        bootstrap=False,
        random_state=SEED,
        warm_start=True,
        n_jobs=6,
    ),
    TransformerModel(
        num_heads=1,
        num_transformer_blocks=1,
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    ),
]

# Spliting dataset
train, test = dataset.filter(lambda traj, _: len(traj) < 200).split(
    0.8, random_state=SEED
)

for model in models:
    # Train the model
    model.train(data=train, cross_validation=5)

    # Evaluate the model on a test dataset
    evaluation = model.evaluate(test)
    evaluations.append(evaluation)

comparison = EvaluationComparison(evaluations)
print(comparison.to_latex())
