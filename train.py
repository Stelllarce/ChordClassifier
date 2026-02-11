import argparse
import sys
import os
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_dataset, split_dataset
from src.models import get_model
from src.utils import print_report, plot_confusion_matrix
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "knn", "lr"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="trained_model.pkl")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir}...")
    features, labels = load_dataset(args.data_dir)
    print(f"Loaded {len(features)} samples")

    X_train, X_test, y_train, y_test = split_dataset(features, labels, test_size=args.test_size)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = get_model(args.model)
    print(f"Training {args.model}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_report(y_test, y_pred)

    fig = plot_confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    joblib.dump(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
