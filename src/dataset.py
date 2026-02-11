import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.features import load_audio, extract_all_features, extract_mel_spectrogram

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]


def load_dataset(data_dir, sr=22050, target_duration=2.0):
    data_dir = Path(data_dir)
    features = []
    labels = []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for audio_file in label_dir.iterdir():
            if audio_file.suffix not in AUDIO_EXTENSIONS:
                continue
            y, sr_loaded = load_audio(str(audio_file), sr=sr, target_duration=target_duration)
            feat = extract_all_features(y, sr_loaded)
            features.append(feat)
            labels.append(label)

    return np.array(features), np.array(labels)


def load_spectrogram_dataset(data_dir, sr=22050, target_duration=2.0, n_mels=128):
    data_dir = Path(data_dir)
    spectrograms = []
    labels = []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for audio_file in label_dir.iterdir():
            if audio_file.suffix not in AUDIO_EXTENSIONS:
                continue
            y, sr_loaded = load_audio(str(audio_file), sr=sr, target_duration=target_duration)
            mel = extract_mel_spectrogram(y, sr_loaded, n_mels=n_mels)
            spectrograms.append(mel)
            labels.append(label)

    return np.array(spectrograms), np.array(labels)


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
