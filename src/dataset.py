"""Load audio folders into feature arrays for training.

Datasets are laid out as <data_dir>/<label>/<file>.wav. This module walks that
layout once and produces whichever representation a model needs, using the exact
same functions the Streamlit app uses at prediction time (src.features) so there
is no train/serve mismatch.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src import config
from src.features import load_audio, chroma_vector, chromagram


def _iter_audio(data_dirs, exclude=None):
    """Yield (path, label) for every audio file under the given folders."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    exclude = exclude or set()

    for data_dir in data_dirs:
        for label_dir in sorted(Path(data_dir).iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.capitalize()
            for audio_file in sorted(label_dir.iterdir()):
                if audio_file.suffix.lower() not in config.AUDIO_EXTENSIONS:
                    continue
                if audio_file.resolve() in exclude:
                    continue
                yield audio_file, label


def load_dataset(data_dirs, feature_type=config.FEATURE_CHROMA_VECTOR, exclude=None):
    """Return (X, labels) using the requested feature representation.

    feature_type = FEATURE_CHROMA_VECTOR -> X shape (n, 24)
    feature_type = FEATURE_CHROMAGRAM    -> X shape (n, 12, N_FRAMES)
    """
    extract = chroma_vector if feature_type == config.FEATURE_CHROMA_VECTOR else chromagram

    features, labels = [], []
    for path, label in _iter_audio(data_dirs, exclude=exclude):
        y, sr = load_audio(str(path))
        features.append(extract(y, sr))
        labels.append(label)

    return np.array(features), np.array(labels)


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
