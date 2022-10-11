from yuca import (
    DecisionTreeModel,
    Featurizer,
    GeoLifeDataset,
    KNeighborsModel,
    RandomForestModel,
    SVMModel,
    features,
)

SEED = 0  # Random seed for reproducibility

# Load Dataset
dataset = GeoLifeDataset()

# Select the desired features to be extracted from the trajectories
featurizer = Featurizer(selected=features.ALL)

# Defining the models
models = [
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
        C=8,
        gamma=5,
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
]


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


for model in models:
    # Train the model
    model.train(data=train, cross_validation=5)

    # Evaluate the model on a test dataset
    evaluation = model.evaluate(test)

    # Print the evaluation
    evaluation.show()
