from tensorflow import keras

from yuca import MnistStrokeDataset
from yuca.models import TransformerModel

SEED = 0  # Random seed for reproducibility

# Load Dataset
dataset = MnistStrokeDataset()

# Define the transformer model
model = TransformerModel(
    head_size=512,
    num_heads=4,
    num_transformer_blocks=4,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
)

# Split into train and test sets
train, test = dataset.split(0.8, random_state=SEED)

# Train the model
model.train(train, epochs=150, batch_size=64)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Print the evaluation
evaluation.show()
