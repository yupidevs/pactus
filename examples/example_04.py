from pactus import Dataset, featurizers
from pactus.models import XGBoostModel

SEED = 0  # Random seed for reproducibility

datasets = [
    Dataset.geolife(),
    Dataset.animals(),
    Dataset.hurdat2(),
    Dataset.cma_bst(),
    Dataset.mnist_stroke(),
    Dataset.uci_pen_digits(),
    Dataset.uci_gotrack(),
    Dataset.uci_characters(),
    Dataset.uci_movement_libras(),
]

featurizer = featurizers.UniversalFeaturizer()

for dataset in datasets:
    print(f"\nDataset: {dataset.name}\n")

    # Split the dataset into train and test
    train, test = dataset.filter(
        lambda traj, _: len(traj) >= 5 and traj.r.delta.norm.sum() > 0
    ).split(
        train_size=0.7,
        random_state=SEED,
    )

    # Select the desired features to be extracted from the trajectories
    featurizer = featurizers.UniversalFeaturizer()

    # Define the model
    model = XGBoostModel(featurizer=featurizer)

    # Evaluate the results
    model.train(data=train, cross_validation=5)
    evaluation = model.evaluate(test)
    evaluation.show()
