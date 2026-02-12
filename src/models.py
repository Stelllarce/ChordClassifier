from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import keras

SKLEARN_MODELS = ["rf", "svm", "knn", "lr"]
KERAS_MODELS = ["mlp", "cnn", "rnn"]
ALL_MODELS = SKLEARN_MODELS + KERAS_MODELS

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

# Pretrained models (VGG, ResNet etc.) were way too slow and performed poorly on
# the small spectrogram inputs, so all the deep learning architectures below are
# built from scratch, as I read they might perform better here

def default_keras_callbacks(patience=8):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def _compile(model):
    # Binary crossentropy since we only have two classes (major / minor)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return _compile(model)


def _conv_block(x, filters):
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return keras.layers.MaxPooling2D((2, 2))(x)


def build_cnn(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Dropout(0.25)(_conv_block(inputs, 32))
    x = keras.layers.Dropout(0.25)(_conv_block(x, 64))
    x = keras.layers.Dropout(0.25)(_conv_block(x, 128))

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return _compile(keras.Model(inputs, outputs))


def build_rnn(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return _compile(keras.Model(inputs, outputs))
