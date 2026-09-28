"""Streamlit app: record or upload a chord, get a major/minor prediction.

All model loading and audio processing is delegated to the shared code in src/,
so this file is just the user interface.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.features import normalize_duration, chroma_vector, chromagram, is_silent
from src import persistence


# --- Inference -----------------------------------------------------------------
@st.cache_resource
def load(model_key):
    return persistence.load_model(model_key)


def features_for(cfg, y, sr):
    """Shape the audio into the input the chosen model expects."""
    if cfg["feature_type"] == config.FEATURE_CHROMA_VECTOR:
        return chroma_vector(y, sr).reshape(1, -1)
    cg = chromagram(y, sr)                      # (12, N_FRAMES)
    if cfg["model_name"] == "cnn":
        return cg[np.newaxis, ..., np.newaxis]  # (1, 12, T, 1)
    return cg.transpose(1, 0)[np.newaxis, ...]  # (1, T, 12) for LSTM/GRU


def predict(model, cfg, le, y, sr):
    """Return (label, confidence) for a prepared clip."""
    X = features_for(cfg, y, sr)
    if cfg["model_type"] == config.MODEL_KERAS:
        prob1 = float(model.predict(X, verbose=0).squeeze())  # P(class index 1)
        idx = int(prob1 > 0.5)
        conf = prob1 if idx == 1 else 1 - prob1
    else:
        idx = int(model.predict(X)[0])
        conf = float(model.predict_proba(X)[0][idx]) if hasattr(model, "predict_proba") else None
    return le.inverse_transform([idx])[0], conf


def load_clip(audio_bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes); tmp.flush(); tmp.close()
        y, sr = librosa.load(tmp.name, sr=config.SR)
    finally:
        os.unlink(tmp.name)
    return normalize_duration(y, sr), sr


# --- Explanation plot ----------------------------------------------------------
def plot_chroma(y, sr, prediction):
    """Average chroma with the root and its major/minor third marked.

    This shows *why* the clip is major or minor: the third above the root.
    """
    mean_chroma = chromagram(y, sr).mean(axis=1)
    root = int(np.argmax(mean_chroma))
    minor_third, major_third = (root + 3) % 12, (root + 4) % 12
    is_major = str(prediction).lower() == "major"

    colors = ["#d9d9d9"] * 12
    colors[root] = "#455a64"
    colors[major_third if is_major else minor_third] = "#ff7043" if is_major else "#4fc3f7"

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(config.NOTE_NAMES, mean_chroma, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Energy")
    ax.set_title(f"Root: {config.NOTE_NAMES[root]}   "
                 f"{'major 3rd: ' + config.NOTE_NAMES[major_third] if is_major else 'minor 3rd: ' + config.NOTE_NAMES[minor_third]}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --- Shared result handler -----------------------------------------------------
def analyse_and_show(audio_bytes, model, cfg, le):
    st.audio(audio_bytes, format="audio/wav")
    with st.spinner("Analysing..."):
        y, sr = load_clip(audio_bytes)
        silent, _ = is_silent(y)
        if silent:
            st.info("No chord detected — the clip is too quiet. Please play a chord.")
            return
        label, conf = predict(model, cfg, le, y, sr)

    is_major = str(label).lower() == "major"
    headline = f"**{label.capitalize()} chord**" + (f" · {conf:.0%} confident" if conf is not None else "")
    (st.success if is_major else st.error)(headline)
    st.pyplot(plot_chroma(y, sr, label))


# --- Layout --------------------------------------------------------------------
st.set_page_config(page_title="Chord Classifier", layout="centered")
st.title("🎵 Chord Classifier")
st.caption("Is a chord **major** or **minor**? Record it or upload a WAV file.")

available = persistence.discover_models()
if not available:
    st.error("No trained models found in models/. Train them first (see notebooks/model_training.ipynb).")
    st.stop()

with st.sidebar:
    st.header("Model")
    keys = list(available.keys())
    choice = st.selectbox(
        "Choose a model",
        keys,
        format_func=lambda k: f"{k.upper()}  (F1 {available[k].get('test_f1', '—')})",
    )
    model, cfg, le = load(choice)
    st.caption(f"Type: {cfg['model_type']} · features: {cfg['feature_type']}")

tab_mic, tab_file = st.tabs(["🎤 Microphone", "📁 Upload"])

with tab_mic:
    rec = st.audio_input("Record about 2 seconds of a chord")
    if rec:
        analyse_and_show(rec.read(), model, cfg, le)

with tab_file:
    up = st.file_uploader("Choose a WAV file", type=["wav"], label_visibility="collapsed")
    if up is not None:
        analyse_and_show(up.read(), model, cfg, le)
