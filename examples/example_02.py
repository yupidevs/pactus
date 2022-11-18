from tensorflow import keras

from pactus import Dataset
from pactus.models import TransformerModel

SEED = 0  # Random seed for reproducibility

# Load Dataset
dataset = Dataset.mnist_stroke()

# Define the transformer model
model = TransformerModel(
    head_size=512,
    num_heads=4,
    num_transformer_blocks=4,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
)

# Split into train and test sets
train, test = dataset.cut(60_000)

# Train the model
checkpoint = keras.callbacks.ModelCheckpoint(
    "partialy_trained_model_mnist_stroke.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)
model.train(train, epochs=150, batch_size=64, checkpoint=checkpoint)

# Evaluate the model on a test dataset
evaluation = model.evaluate(test)

# Print the evaluation
evaluation.show()
