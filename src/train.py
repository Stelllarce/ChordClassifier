"""Train the whole model zoo and save each model into models/<name>/.

Run locally:      python -m src.train            (trains everything)
                  python -m src.train --quick    (fewer epochs, for a fast check)
From Colab:       import src.train as train; train.main(epochs=80)

All heavy logic lives here so the notebook only has to download data and call
`main()`. Features and augmentation come from src.features, so training and the
live app process audio identically.
"""

import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src import config
from src.features import (load_audio, chroma_vector, chromagram,
                          augment_vectors, augment_chromagrams)
from src import models, persistence


# --- Data ----------------------------------------------------------------------
def load_features(data_dirs):
    """Load every clip once and return both feature views + labels.

    Xvec:  (n, 24)          aggregated chroma for sklearn + MLP
    Xgram: (n, 12, N_FRAMES) chromagram image for CNN / LSTM / GRU
    """
    from src.dataset import _iter_audio
    vecs, grams, labels = [], [], []
    for path, label in _iter_audio(data_dirs):
        y, sr = load_audio(str(path))
        vecs.append(chroma_vector(y, sr))
        grams.append(chromagram(y, sr))
        labels.append(label)
    return np.array(vecs), np.array(grams), np.array(labels)


# --- Evaluation helpers --------------------------------------------------------
# Linear models can't represent transposition-invariant chord quality, so mixing
# in all 12 key-rotations only muddies them; they train on the original clips.
NO_AUGMENT = {"lr"}


def _cv_f1_sklearn(name, X, y, augment, n_splits=5):
    """Cross-validated F1, augmenting only inside each training fold (no leakage)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, val in skf.split(X, y):
        Xt, yt = augment_vectors(X[tr], y[tr]) if augment else (X[tr], y[tr])
        pipe = make_pipeline(StandardScaler(), models.get_sklearn_model(name))
        pipe.fit(Xt, yt)
        scores.append(f1_score(y[val], pipe.predict(X[val]), average="weighted"))
    return float(np.mean(scores))


def _eval(y_true, y_pred):
    return (f1_score(y_true, y_pred, average="weighted"),
            accuracy_score(y_true, y_pred))


# --- Training per family -------------------------------------------------------
def _train_sklearn(name, Xtr, ytr, Xte, yte):
    augment = name not in NO_AUGMENT
    cv = _cv_f1_sklearn(name, Xtr, ytr, augment)
    Xt, yt = augment_vectors(Xtr, ytr) if augment else (Xtr, ytr)
    pipe = make_pipeline(StandardScaler(), models.get_sklearn_model(name))
    pipe.fit(Xt, yt)
    f1, acc = _eval(yte, pipe.predict(Xte))
    cfg = persistence.make_config(name, config.MODEL_SKLEARN, config.FEATURE_CHROMA_VECTOR,
                                  config.LABELS, augment=augment, cv_f1=cv, test_f1=f1, test_acc=acc)
    return pipe, cfg, f1


def _train_keras(name, build_fn, Xtr, ytr, Xte, yte, epochs, batch_size=32):
    model = build_fn(Xtr.shape[1:])
    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, validation_split=0.15,
              verbose=0, callbacks=models.default_keras_callbacks(patience=10))
    y_pred = (model.predict(Xte, verbose=0).squeeze() >= 0.5).astype(int)
    f1, acc = _eval(yte, y_pred)
    cfg = persistence.make_config(name, config.MODEL_KERAS, config.FEATURE_CHROMAGRAM,
                                  config.LABELS, test_f1=f1, test_acc=acc)
    if name == "mlp":  # MLP uses the flat vector, not the chromagram
        cfg["feature_type"] = config.FEATURE_CHROMA_VECTOR
    return model, cfg, f1


# --- Orchestration -------------------------------------------------------------
def main(data_dirs=None, epochs=80, models_to_train=None, quick=False):
    if data_dirs is None:
        data_dirs = [config.DATA_DIR / "shords_dataset", config.DATA_DIR / "Audio_Files"]
    if quick:
        epochs = 8
    train_set = set(models_to_train) if models_to_train else None

    print("Loading audio and extracting features...")
    Xvec, Xgram, labels = load_features(data_dirs)
    le = LabelEncoder(); y = le.fit_transform(labels)
    print(f"  {len(y)} clips | classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # One split shared by every model (indices, so both feature views align).
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    ytr, yte = y[idx_tr], y[idx_te]

    results = {}

    def want(n):
        return train_set is None or n in train_set

    # Classical + MLP use the flat chroma vector.
    Xv_tr, Xv_te = Xvec[idx_tr], Xvec[idx_te]
    for name in config.SKLEARN_MODELS:
        if not want(name):
            continue
        print(f"Training {name} (sklearn)...")
        model, cfg, f1 = _train_sklearn(name, Xv_tr, ytr, Xv_te, yte)
        persistence.save_model(model, cfg, le)
        results[name] = f1

    if want("mlp"):
        print("Training mlp (keras)...")
        model, cfg, f1 = _train_keras("mlp", lambda s: models.build_mlp(s[0]),
                                      Xv_tr, ytr, Xv_te, yte, epochs)
        persistence.save_model(model, cfg, le)
        results["mlp"] = f1

    # CNN/LSTM/GRU use the chromagram image; augment the training split.
    Xg_tr_a, ytr_a = augment_chromagrams(Xgram[idx_tr], ytr)
    Xg_te = Xgram[idx_te]
    keras_seq = {
        "cnn": (models.build_cnn, Xg_tr_a[..., np.newaxis], Xg_te[..., np.newaxis]),
        "lstm": (models.build_lstm, Xg_tr_a.transpose(0, 2, 1), Xg_te.transpose(0, 2, 1)),
        "gru": (models.build_gru, Xg_tr_a.transpose(0, 2, 1), Xg_te.transpose(0, 2, 1)),
    }
    for name, (build_fn, Xt, Xe) in keras_seq.items():
        if not want(name):
            continue
        print(f"Training {name} (keras)...")
        model, cfg, f1 = _train_keras(name, build_fn, Xt, ytr_a, Xe, yte, epochs)
        persistence.save_model(model, cfg, le)
        results[name] = f1

    print("\n=== Test F1 (weighted) ===")
    for name, f1 in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:5s} {f1:.4f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--quick", action="store_true", help="few epochs for a fast smoke test")
    ap.add_argument("--models", nargs="*", help="subset of model names to train")
    args = ap.parse_args()
    main(epochs=args.epochs, models_to_train=args.models, quick=args.quick)
