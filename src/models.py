"""Model definitions for the chord-quality (major vs minor) classifiers.

Two families, as the assignment requires:
  - classical scikit-learn models on aggregated chroma vectors (rf/svm/knn/lr);
  - small Keras networks: an MLP on the same vectors, a 2D CNN on the chromagram
    image, and LSTM/GRU recurrent nets over the chroma frame sequence.

All nets are deliberately compact so they train quickly (seconds to a couple of
minutes on a Colab GPU); the heavy lifting comes from the features + rotation
augmentation, not from network size.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import keras

from src import config

# --- Classical models ----------------------------------------------------------
SKLEARN_DEFAULTS = {
    "rf": dict(n_estimators=300, max_depth=None, min_samples_split=4,
               class_weight="balanced", random_state=42),
    "svm": dict(kernel="rbf", C=10.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=42),
    "knn": dict(n_neighbors=7, weights="distance", metric="minkowski"),
    "lr": dict(C=1.0, max_iter=1000, solver="lbfgs",
               class_weight="balanced", random_state=42),
}


def get_sklearn_model(name="rf", **kwargs):
    models = {
        "rf": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**{**SKLEARN_DEFAULTS.get(name, {}), **kwargs})


# --- Keras helpers -------------------------------------------------------------
def default_keras_callbacks(patience=8):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def _compile(model):
    # Binary crossentropy: two classes (major / minor) with a sigmoid output.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# --- MLP on the 24-dim chroma vector ------------------------------------------
def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return _compile(model)


# --- CNN on the chromagram image (12 pitch classes x time) ---------------------
def build_cnn(input_shape):
    """input_shape = (12, N_FRAMES, 1)."""
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64):
        x = keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, outputs))


# --- Recurrent nets over the chroma frame sequence -----------------------------
def _build_rnn(input_shape, cell):
    """input_shape = (N_FRAMES, 12). `cell` is keras.layers.LSTM or GRU."""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Bidirectional(cell(64, return_sequences=True))(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Bidirectional(cell(32))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, outputs))


def build_lstm(input_shape):
    return _build_rnn(input_shape, keras.layers.LSTM)


def build_gru(input_shape):
    return _build_rnn(input_shape, keras.layers.GRU)
