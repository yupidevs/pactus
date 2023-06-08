from typing import Tuple


from pactus import Dataset, featurizers
from pactus.dataset.dataset import Data
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

def dataset_splitter(ds: Data) -> Tuple[Data, Data]:
    if ds.dataset_name == "geolife":
        use_classes = {"car", "taxi-bus", "walk", "bike", "subway", "train"}
        return (
            ds.filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)
            .map(lambda _, lbl: (_, "taxi-bus" if lbl in ("bus", "taxi") else lbl))
            .filter(lambda _, lbl: lbl in use_classes)
            .split(train_size=0.7, random_state=SEED)
        )
    if ds.dataset_name == "mnist_stroke":
        ds = ds.take(10_000)
    return ds.filter(
        lambda traj, _: len(traj) >= 5 and traj.r.delta.norm.sum() > 0
    ).split(train_size=0.7, random_state=SEED)


featurizer = featurizers.UniversalFeaturizer()

for dataset in datasets:
    print(f"\nDataset: {dataset.name}\n")

    # Split the dataset into train and test
    train, test = dataset_splitter(dataset)

    # Select the desired features to be extracted from the trajectories
    featurizer = featurizers.UniversalFeaturizer()

    # Define the model
    model = XGBoostModel(featurizer=featurizer)

    # Evaluate the results
    model.train(data=train, cross_validation=5)
    evaluation = model.evaluate(test)
    evaluation.show()
