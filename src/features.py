"""Audio -> feature conversion, shared by the app and the training notebook.

The task is major-vs-minor, which is decided by one interval (the third). So the
features here are built around *chroma* (energy per pitch class C..B), which is
exactly where that information lives. Timbre features like MFCC are deliberately
left out of the model inputs (they mostly describe the instrument, not the chord)
and kept only for exploratory analysis.
"""

import numpy as np
import librosa

from src import config


# --- Duration / loading --------------------------------------------------------
def normalize_duration(y, sr, target_duration=config.TARGET_DURATION):
    """Pad with silence or truncate so every clip is exactly target_duration."""
    target_length = int(sr * target_duration)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    return y


def load_audio(path, sr=config.SR, target_duration=config.TARGET_DURATION):
    y, sr = librosa.load(path, sr=sr)
    if target_duration is not None:
        y = normalize_duration(y, sr, target_duration=target_duration)
    return y, sr


# --- Chroma building blocks ----------------------------------------------------
def chromagram(y, sr, hop_length=config.HOP_LENGTH):
    """2D Constant-Q chroma: shape (12 pitch classes, time frames).

    CQT (Constant-Q Transform) chroma is preferred over the STFT version because
    its log-frequency spacing lines up with musical pitch, giving a cleaner
    pitch-class picture of the chord.
    """
    return librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=config.N_CHROMA, hop_length=hop_length)


def chroma_vector(y, sr):
    """Aggregate a clip into a flat feature vector: [mean(12), std(12)] = 24 dims.

    The layout is two stacked 12-pitch-class blocks so it can be transposed with
    rotate_chroma_vector (see below).
    """
    chroma = chromagram(y, sr)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])


# --- Pitch-rotation augmentation ----------------------------------------------
# A chord shifted up N semitones keeps its quality (major stays major). Rolling
# the pitch axis of a chroma representation is therefore a free, label-preserving
# way to create the 11 other keys, which teaches models to ignore the root note.
def rotate_chromagram(chroma, semitones):
    """Roll the 12 pitch-class rows by `semitones` (transpose the chord)."""
    return np.roll(chroma, semitones, axis=0)


def rotate_chroma_vector(vec, semitones, n_chroma=config.N_CHROMA):
    """Roll every 12-length block of an aggregated chroma vector."""
    blocks = vec.reshape(-1, n_chroma)
    return np.roll(blocks, semitones, axis=1).reshape(-1)


def augment_vectors(X, y, n_chroma=config.N_CHROMA):
    """Expand a (samples, features) set with all 12 transpositions (x12 data)."""
    X_aug, y_aug = [], []
    for vec, label in zip(X, y):
        for k in range(n_chroma):
            X_aug.append(rotate_chroma_vector(vec, k, n_chroma))
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)


def augment_chromagrams(X, y):
    """Expand a (samples, 12, frames) set with all 12 transpositions (x12 data)."""
    X_aug, y_aug = [], []
    for chroma, label in zip(X, y):
        for k in range(config.N_CHROMA):
            X_aug.append(rotate_chromagram(chroma, k))
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)


# --- Silence / tuning checks (noisy-data handling) -----------------------------
def is_silent(y, threshold=config.RMS_SILENCE_THRESHOLD):
    """True when the clip is too quiet to contain a chord (RMS loudness gate)."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    return rms < threshold, rms


def estimate_tuning_deviation(y, sr):
    """How far the recording is from standard 440 Hz tuning, in semitone fraction.

    ~0 means in tune; e.g. +0.5 means tuned a quarter-tone sharp. Useful for
    spotting recordings that deviate from the standard reference pitch.
    """
    return float(librosa.estimate_tuning(y=y, sr=sr))


# --- MFCC (exploratory analysis only, not used as a model input) ---------------
def extract_mfcc(y, sr, n_mfcc=13, hop_length=config.HOP_LENGTH):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return np.mean(mfcc, axis=1)
