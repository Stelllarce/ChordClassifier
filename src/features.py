import numpy as np
import librosa

"""Functions to work with features. Needed for the application"""

def normalize_duration(y, sr, target_duration=2.0):
    target_length = int(sr * target_duration)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    return y


def extract_chroma(y, sr, n_chroma=12, hop_length=512):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    return np.mean(chroma, axis=1)


def extract_mfcc(y, sr, n_mfcc=13, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return np.mean(mfcc, axis=1)


def extract_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def extract_all_features(y, sr):
    chroma = extract_chroma(y, sr)
    mfcc = extract_mfcc(y, sr)
    return np.concatenate([chroma, mfcc])


def load_audio(path, sr=22050, target_duration=2.0):
    y, sr = librosa.load(path, sr=sr)
    if target_duration is not None:
        y = normalize_duration(y, sr, target_duration=target_duration)
    return y, sr


def extract_windowed_features(y, sr, window_sec=0.5, overlap=0.5):
    window_samples = int(window_sec * sr)
    hop_samples = int(window_samples * (1 - overlap))

    window_feats = []
    start = 0
    while start + window_samples <= len(y):
        segment = y[start:start + window_samples]
        feat = np.concatenate([extract_chroma(segment, sr), extract_mfcc(segment, sr)])
        window_feats.append(feat)
        start += hop_samples

    if not window_feats:
        window_feats.append(np.concatenate([extract_chroma(y, sr), extract_mfcc(y, sr)]))

    window_feats = np.array(window_feats)
    return np.concatenate([np.mean(window_feats, axis=0), np.std(window_feats, axis=0)])
