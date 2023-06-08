from tensorflow import keras

from pactus import Dataset, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    LSTMModel,
    RandomForestModel,
    SVMModel,
    TransformerModel, # TODO: Falta XGBoost
)

SEED = 0 # TODO: Use this for reproducibility

dataset = Dataset.mnist_stroke()
train, test = dataset.cut(60_000)

featurizer = featurizers.UniversalFeaturizer()
vectorized_models = [
    RandomForestModel(
        featurizer=featurizer,
        max_features=16,
        n_estimators=200,
        bootstrap=False,
        warm_start=True,
        n_jobs=6,
    ),
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
    ),
]


transformer = TransformerModel(
    head_size=512,
    num_heads=4,
    num_transformer_blocks=4,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
)

lstm = LSTMModel(
    loss="sparse_categorical_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"],
)

# Train and evaluate vectorized models
for model in vectorized_models:
    print(f"\nModel: {model.name}\n")
    model.train(train, cross_validation=5)
    evaluation = model.evaluate(test)
    evaluation.show()

# Train and evaluate LSTM model
checkpoint = keras.callbacks.ModelCheckpoint(
    "partially_trained_model_lstm_mnist_stroke.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)
lstm.train(train, dataset, epochs=20, checkpoint=checkpoint)
evaluation = lstm.evaluate(test)
evaluation.show()

# Train and evaluate Transformer model
checkpoint = keras.callbacks.ModelCheckpoint(
    "partially_trained_model_transformer_mnist_stroke.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)
transformer.train(train, dataset, epochs=150, checkpoint=checkpoint)
evaluation = transformer.evaluate(test)
evaluation.show()
