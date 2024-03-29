from pactus import Dataset, featurizers
from pactus.models import RandomForestModel

SEED = 0  # Random seed for reproducibility

# Load Dataset
dataset = Dataset.geolife()

# Select the desired features to be extracted from the trajectories
featurizer = featurizers.UniversalFeaturizer()

# Defining the models
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
    # Remove short and poorly time sampled trajectories
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
