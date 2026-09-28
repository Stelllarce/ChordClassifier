"""Save/load models and their config in one place.

Previously the training notebook wrote models one way and app.py read them
another way, so the little JSON "config" describing each model was defined twice.
This module is the single definition of that contract.

Each model lives in models/<name>/ with:
  - config.json        metadata + how to feed the model (see make_config)
  - label_encoder.pkl  maps the predicted 0/1 back to "Major"/"Minor"
  - model.pkl          (sklearn) or model.keras (keras)
"""

import json
from pathlib import Path

import joblib

from src import config


def make_config(model_name, model_type, feature_type, labels,
                augment=True, cv_f1=None, test_f1=None, test_acc=None):
    """Build the standard config dict saved next to every model."""
    return {
        "model_name": model_name,
        "model_type": model_type,        # config.MODEL_SKLEARN / MODEL_KERAS
        "feature_type": feature_type,    # config.FEATURE_CHROMA_VECTOR / FEATURE_CHROMAGRAM
        "labels": list(labels),
        "augment": augment,
        "cv_f1": _round(cv_f1),
        "test_f1": _round(test_f1),
        "test_acc": _round(test_acc),
    }


def _round(x, ndigits=4):
    return round(float(x), ndigits) if x is not None else None


def save_model(model, cfg, label_encoder, model_dir=None):
    """Persist a trained model, its config, and the label encoder."""
    model_dir = _resolve_dir(model_dir if model_dir is not None else cfg["model_name"])
    model_dir.mkdir(parents=True, exist_ok=True)

    if cfg["model_type"] == config.MODEL_KERAS:
        model.save(str(model_dir / "model.keras"))
    else:
        joblib.dump(model, str(model_dir / "model.pkl"))

    joblib.dump(label_encoder, str(model_dir / "label_encoder.pkl"))
    with open(model_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return model_dir


def load_model(model_key):
    """Load (model, config, label_encoder) for a model directory name.

    Returns (None, None, None) if the directory has no config.json.
    """
    model_dir = _resolve_dir(model_key)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None, None, None

    with open(cfg_path) as f:
        cfg = json.load(f)
    label_encoder = joblib.load(str(model_dir / "label_encoder.pkl"))

    if cfg["model_type"] == config.MODEL_KERAS:
        import keras
        model = keras.models.load_model(str(model_dir / "model.keras"))
    else:
        model = joblib.load(str(model_dir / "model.pkl"))

    return model, cfg, label_encoder


def discover_models(models_dir=config.MODEL_DIR):
    """Return {name: config} for every models/<name>/ that has a config.json."""
    available = {}
    if not models_dir.exists():
        return available
    for d in sorted(models_dir.iterdir()):
        cfg_path = d / "config.json"
        if d.is_dir() and cfg_path.exists():
            with open(cfg_path) as f:
                available[d.name] = json.load(f)
    return available


def _resolve_dir(model_key_or_dir):
    """A bare name like "rf" resolves under models/; a real path is used as-is."""
    p = Path(model_key_or_dir)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return config.MODEL_DIR / str(model_key_or_dir)
