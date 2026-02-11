import numpy as np
import librosa


def normalize_duration(y, sr, target_duration=2.0):
    """Normalize audio to a fixed duration by truncating or zero-padding.
    
    Args:
        y: Audio time series
        sr: Sample rate
        target_duration: Target duration in seconds (default 2.0)
    
    Returns:
        y_fixed: Audio array of exactly target_duration * sr samples
    """
    target_length = int(sr * target_duration)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
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


def load_audio(path, sr=22050, duration=None, target_duration=2.0):
    """Load audio and optionally normalize to a fixed duration.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        duration: Max duration to load (librosa parameter, None = full file)
        target_duration: If set, normalize audio to this duration in seconds.
                        Set to None to skip normalization.
    """
    y, sr = librosa.load(path, sr=sr, duration=duration)
    if target_duration is not None:
        y = normalize_duration(y, sr, target_duration=target_duration)
    return y, sr
