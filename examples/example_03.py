from tensorflow import keras

from pactus import Dataset, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    LSTMModel,
    RandomForestModel,
    SVMModel,
    TransformerModel,
    XGBoostModel,
)

SEED = 0  # Random seed for reproducibility

dataset = Dataset.uci_characters()
train, test = dataset.split(.8, random_state=SEED)

featurizer = featurizers.UniversalFeaturizer()
vectorized_models = [
    RandomForestModel(
        featurizer=featurizer,
        max_features=16,
        n_estimators=200,
        bootstrap=False,
        warm_start=True,
        n_jobs=6,
        random_state=SEED,
    ),
    KNeighborsModel(
        featurizer=featurizer,
        n_neighbors=7,
    ),
    DecisionTreeModel(
        featurizer=featurizer,
        max_depth=7,
        random_state=SEED,
    ),
    SVMModel(
        featurizer=featurizer,
        random_state=SEED,
    ),
    XGBoostModel(
        featurizer=featurizer,
        random_state=SEED,
    ),
]


transformer = TransformerModel(
    head_size=512,
    num_heads=4,
    num_transformer_blocks=4,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    random_state=SEED,
)

lstm = LSTMModel(
    loss="sparse_categorical_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"],
    random_state=SEED,
)

# Train and evaluate vectorized models
for model in vectorized_models:
    print(f"\nModel: {model.name}\n")
    model.train(train, cross_validation=5)
    evaluation = model.evaluate(test)
    evaluation.show()

# Train and evaluate LSTM model
checkpoint = keras.callbacks.ModelCheckpoint(
    f"partially_trained_model_lstm_{dataset.name}.h5",
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
    f"partially_trained_model_transformer_{dataset.name}.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)
transformer.train(train, dataset, epochs=150, checkpoint=checkpoint)
evaluation = transformer.evaluate(test)
evaluation.show()
