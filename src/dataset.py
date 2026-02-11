import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.features import load_audio, extract_all_features


def load_dataset(data_dir, sr=22050, duration=None, target_duration=2.0):
    """Load audio dataset from directory structure.
    
    Args:
        data_dir: Root directory containing subdirectories per class
        sr: Sample rate
        duration: Max duration to load per file (librosa parameter)
        target_duration: Normalize all samples to this duration in seconds.
                        Set to None to skip normalization. Default 2.0s.
    """
    data_dir = Path(data_dir)
    features = []
    labels = []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for audio_file in label_dir.iterdir():
            if audio_file.suffix not in [".wav", ".mp3", ".flac", ".ogg"]:
                continue
            y, sr_loaded = load_audio(str(audio_file), sr=sr, duration=duration,
                                      target_duration=target_duration)
            feat = extract_all_features(y, sr_loaded)
            features.append(feat)
            labels.append(label)

    return np.array(features), np.array(labels)


def split_dataset(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
