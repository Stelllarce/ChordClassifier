import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def plot_class_distribution(labels, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(unique, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.tight_layout()
    return fig


def plot_waveform(y, sr, title="Waveform", figsize=(10, 3)):
    import librosa.display
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_spectrogram(mel_db, sr, hop_length=512, title="Mel Spectrogram", figsize=(10, 4)):
    import librosa.display
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig
