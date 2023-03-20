from tensorflow import keras

from pactus import Dataset, featurizers
from pactus.dataset.dataset import Data
from pactus.models import TransformerModel

SEED = 0  # Random seed for reproducibility

datasets = [
    Dataset.geolife(),
    Dataset.animals(),
    Dataset.hurdat2(),
    Dataset.cma_bst(),
    Dataset.mnist_stroke(),
    Dataset.uci_gotrack(),
    Dataset.uci_characters(),
    Dataset.uci_movement_libras(),
]


def dataset_splitter(ds: Data) -> tuple[Data, Data]:
    if ds.dataset.name == "geolife":
        use_classes = {"car", "taxi-bus", "walk", "bike", "subway", "train"}
        return (
            ds.filter(lambda traj, _: len(traj) > 10 and traj.dt < 8)
            .map(lambda _, lbl: (_, "taxi-bus" if lbl in ("bus", "taxi") else lbl))
            .filter(lambda _, lbl: lbl in use_classes)
            .split(train_size=0.7, random_state=SEED)
        )
    if ds.dataset.name == "mnist_stroke":
        ds = ds.take(10_000)
    return ds.filter(
        lambda traj, _: len(traj) >= 5 and traj.r.delta.norm.sum() > 0
    ).split(train_size=0.7, random_state=SEED)


featurizer = featurizers.UniversalFeaturizer()

for dataset in datasets:
    print(f"\nDataset: {dataset.name}\n")

    # Split the dataset into train and test
    train, test = dataset_splitter(dataset)

    # Define the model
    model = TransformerModel(
        head_size=512,
        num_heads=4,
        num_transformer_blocks=4,
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    )

    model.train(train, epochs=150, batch_size=64)
    evaluation = model.evaluate(test)
    evaluation.show()
