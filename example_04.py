from typing import List

from yuca import (
    DecisionTreeModel,
    Evaluation,
    EvaluationComparison,
    Featurizer,
    GeoLifeDataset,
    KNeighborsModel,
    LangevinDataset,
    RandomForestModel,
    SVMModel,
    features,
)

SEED = 0  # Random seed for reproducibility

evaluations: List[Evaluation] = []

# Load Dataset
datasets = [LangevinDataset(), GeoLifeDataset()]

# Select the desired features to be extracted from the trajectories
featurizer = Featurizer(selected=features.ALL)

# Defining the model
for dataset in datasets:
    models = [
        KNeighborsModel(
            featurizer=featurizer,
            n_neighbors=7,
        ),
        DecisionTreeModel(
            featurizer=featurizer,
            max_depth=7,
        ),
        # SVMModel(
        #     featurizer=featurizer,
        #     C=8,
        #     gamma=5,
        # ),
        RandomForestModel(
            featurizer=featurizer,
            max_features=16,
            n_estimators=200,
            bootstrap=False,
            random_state=SEED,
            warm_start=True,
            n_jobs=6,
        ),
    ]

    # Spliting dataset
    train, test = dataset.split(0.8, random_state=SEED)

    for model in models:
        # Train the model
        model.train(data=train, cross_validation=5)

        # Evaluate the model on a test dataset
        evaluation = model.evaluate(test)

        evaluations.append(evaluation)


comparison = EvaluationComparison(evaluations)
print(comparison.to_latex())
