from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import keras


SKLEARN_MODELS = ["rf", "svm", "knn", "lr"]
KERAS_MODELS = ["mlp", "cnn", "rnn"]
ALL_MODELS = SKLEARN_MODELS + KERAS_MODELS


def get_sklearn_model(name="rf", **kwargs):
    models = {
        "rf": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    if input_shape[-1] == 1:
        x = keras.layers.Concatenate()([inputs, inputs, inputs])
    else:
        x = inputs

    x = keras.layers.Resizing(128, 128)(x)

    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3),
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_rnn(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Reshape((*input_shape, 1))(inputs)
    x = keras.layers.Concatenate()([x, x, x])
    x = keras.layers.Resizing(128, 128)(x)

    base_model = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3),
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
