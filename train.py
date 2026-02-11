import argparse
import json
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import LabelEncoder
from src.dataset import load_dataset, load_spectrogram_dataset, split_dataset
from src.models import get_sklearn_model, build_mlp, build_cnn, build_rnn
from src.models import SKLEARN_MODELS, KERAS_MODELS, ALL_MODELS
from src.utils import print_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="rf", choices=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    is_keras = args.model in KERAS_MODELS
    use_spectrogram = args.model in ["cnn", "rnn"]

    print(f"Loading dataset from {args.data_dir}...")
    if use_spectrogram:
        X, labels = load_spectrogram_dataset(args.data_dir)
    else:
        X, labels = load_dataset(args.data_dir)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"Loaded {len(X)} samples, classes: {le.classes_.tolist()}")

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=args.test_size)

    if args.model == "cnn":
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    elif args.model == "rnn":
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    print(f"Training {args.model}...")
    if args.model == "mlp":
        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                  validation_split=0.15, verbose=1)
    elif args.model == "cnn":
        model = build_cnn(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                  validation_split=0.15, verbose=1)
    elif args.model == "rnn":
        model = build_rnn(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                  validation_split=0.15, verbose=1)
    else:
        model = get_sklearn_model(args.model)
        model.fit(X_train, y_train)

    if is_keras:
        y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    print_report(le.inverse_transform(y_test), le.inverse_transform(y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)

    config = {
        "model_name": args.model,
        "model_type": "keras" if is_keras else "sklearn",
        "feature_type": "spectrogram" if use_spectrogram else "features",
        "labels": le.classes_.tolist(),
    }

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    if is_keras:
        model.save(os.path.join(MODEL_DIR, "model.keras"))
    else:
        joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))

    print(f"Model saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
