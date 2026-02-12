import streamlit as st
import numpy as np
import librosa
import librosa.display
import keras
import joblib
import json
import os
import sys
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import LabelEncoder
from src.features import extract_all_features, extract_mel_spectrogram, normalize_duration, extract_windowed_features

SR = 22050
TARGET_DURATION = 2.0
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIRS = [
    PROJECT_ROOT / "data" / "shords_dataset",
    PROJECT_ROOT / "data" / "Audio_Files",
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def is_chord(y, sr, rms_threshold=0.005):
    rms = np.sqrt(np.mean(y ** 2))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    if rms < rms_threshold or np.max(chroma_mean) == 0:
        return False, chroma_mean

    return True, chroma_mean


def process_audio_bytes(audio_bytes, sr=SR):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        y, sr_out = librosa.load(tmp.name, sr=sr)
    finally:
        os.unlink(tmp.name)
    return y, sr_out


def classify_audio(y, sr, model, config, le):
    y = normalize_duration(y, sr, target_duration=TARGET_DURATION)
    is_chord_flag, chroma_mean = is_chord(y, sr)

    if not is_chord_flag:
        return None, "No audio detected. Please play a chord.", chroma_mean

    if config["feature_type"] == "spectrogram":
        mel = extract_mel_spectrogram(y, sr)
        if config["model_name"] == "cnn":
            X = mel[np.newaxis, ..., np.newaxis]
        else:
            X = mel.T[np.newaxis, ...]
    else:
        if config.get("window_sec"):
            X = extract_windowed_features(y, sr, config["window_sec"], config.get("overlap", 0.5)).reshape(1, -1)
        else:
            X = extract_all_features(y, sr).reshape(1, -1)

    if config["model_type"] == "keras":
        prob = model.predict(X, verbose=0).flatten()[0]
        pred_idx = int(prob > 0.5)
    else:
        pred_idx = model.predict(X)[0]

    prediction = le.inverse_transform([pred_idx])[0]
    return prediction, None, chroma_mean


def plot_chroma(chroma_mean):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    colours = ["#4fc3f7" if v < np.max(chroma_mean) * 0.45 else "#ff7043"
               for v in chroma_mean]
    ax.bar(NOTE_NAMES, chroma_mean, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Energy")
    ax.set_title("Chroma Profile")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4fc3f7")
    ax.set_title("Waveform")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def discover_models():
    available = {}
    for d in sorted(MODEL_DIR.iterdir()):
        cfg_path = d / "config.json"
        if d.is_dir() and cfg_path.exists():
            with open(cfg_path) as f:
                available[d.name] = json.load(f)
    return available


@st.cache_resource
def load_model(model_key):
    model_dir = MODEL_DIR / model_key
    config_path = model_dir / "config.json"
    le_path = model_dir / "label_encoder.pkl"

    if not config_path.exists():
        return None, None, None

    with open(config_path) as f:
        config = json.load(f)
    le = joblib.load(str(le_path))

    if config["model_type"] == "keras":
        model = keras.models.load_model(str(model_dir / "model.keras"))
    else:
        model = joblib.load(str(model_dir / "model.pkl"))

    return model, config, le


def show_result(prediction, warning, chroma_mean, y, sr):
    if warning:
        st.info(warning)
    else:
        label = str(prediction).capitalize()
        if label.lower() == "major":
            st.success(f"**{label} chord** detected")
        else:
            st.error(f"**{label} chord** detected")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_waveform(y, sr))
    with col2:
        st.pyplot(plot_chroma(chroma_mean))


st.set_page_config(page_title="Chord Classifier", layout="centered")
# Forgot my css so I needed an AI to make this look decent.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {min-width:260px; max-width:320px;}
    div[data-testid="stFileUploader"] > div {border: 2px dashed #4fc3f7; border-radius:1rem; padding:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    available_models = discover_models()

    if not available_models:
        st.error("No trained models found.")
        model, config, le = None, None, None
    else:
        options = list(available_models.keys())
        labels = [
            f"{k.upper()}  (F1: {v.get('test_f1', 'N/A')})"
            for k, v in available_models.items()
        ]
        selected_idx = st.selectbox(
            "Model",
            range(len(options)),
            format_func=lambda i: labels[i],
        )
        selected_key = options[selected_idx]
        model, config, le = load_model(selected_key)

        if model is None:
            st.error("Failed to load model.")
        else:
            f1 = available_models[selected_key].get("test_f1")
            st.success(f"**{selected_key.upper()}** loaded — F1: {f1}")

    st.divider()
    st.caption("Chord Classifier - Major / Minor")

st.title("Chord Classifier")
st.caption("Detect whether a chord is **Major** or **Minor** from microphone or a WAV file.")

if model is None:
    st.error("No model available.")
    st.stop()

tab_mic, tab_file = st.tabs(["Microphone", "Upload file"])

with tab_mic:
    st.markdown("Record a short clip (about 2 seconds) of a chord on your instrument.")

    audio_input = st.audio_input("Record a chord")

    if audio_input:
        audio_bytes = audio_input.read()
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Analysing..."):
            y, sr = process_audio_bytes(audio_bytes, SR)
            prediction, warning, chroma_mean = classify_audio(y, sr, model, config, le)
        show_result(prediction, warning, chroma_mean, y, sr)

with tab_file:
    st.markdown("Drag-and-drop or browse for a .wav file.")

    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=["wav"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Analysing..."):
            y, sr = process_audio_bytes(audio_bytes, SR)
            prediction, warning, chroma_mean = classify_audio(y, sr, model, config, le)
        show_result(prediction, warning, chroma_mean, y, sr)
