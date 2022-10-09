from yuca import MnistStrokeDataset
from yuca.models import TransformerModel

SEED = 0  # Random seed for reproducibility

# Load Dataset
dataset = MnistStrokeDataset()

# Define the transformer model
model = TransformerModel()

# Split into train and test sets
train, test = dataset.split(0.8, random_state=SEED)

# Train the model
model.train(train, epochs=10)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Print the evaluation
evaluation.show()
