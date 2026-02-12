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
from src.dataset import load_dataset, split_dataset
from src.models import get_sklearn_model

SR = 22050
TARGET_DURATION = 2.0
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIRS = [
    PROJECT_ROOT / "data" / "shords_dataset",
    PROJECT_ROOT / "data" / "Audio_Files",
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def is_chord(y, sr, chroma_threshold=0.45, min_chord_notes=3, rms_threshold=0.005):
    rms = np.sqrt(np.mean(y ** 2))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    if rms < rms_threshold or np.max(chroma_mean) == 0:
        return False, "silence", chroma_mean

    chroma_norm = chroma_mean / np.max(chroma_mean)
    strong_bins = int(np.sum(chroma_norm >= chroma_threshold))

    if strong_bins >= min_chord_notes:
        return True, "chord", chroma_mean
    return False, "single_note", chroma_mean


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
    is_chord_flag, audio_type, chroma_mean = is_chord(y, sr)

    if audio_type == "silence":
        return None, "No audio detected. Please play a chord.", chroma_mean

    if not is_chord_flag:
        return None, "Single note detected. Please play a chord.", chroma_mean

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


@st.cache_resource
def load_model():
    config_path = MODEL_DIR / "config.json"
    le_path = MODEL_DIR / "label_encoder.pkl"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        le = joblib.load(str(le_path))
        if config["model_type"] == "keras":
            model = keras.models.load_model(str(MODEL_DIR / "model.keras"))
        else:
            model = joblib.load(str(MODEL_DIR / "model.pkl"))
        return model, config, le, "loaded"

    for data_dir in DATA_DIRS:
        if data_dir.exists() and any(data_dir.iterdir()):
            X, labels = load_dataset(str(data_dir))
            if len(X) > 0:
                le = LabelEncoder()
                y = le.fit_transform(labels)
                X_train, X_test, y_train, y_test = split_dataset(X, y)
                model = get_sklearn_model("rf")
                model.fit(X_train, y_train)

                config = {
                    "model_name": "rf",
                    "model_type": "sklearn",
                    "feature_type": "features",
                    "labels": le.classes_.tolist(),
                }

                os.makedirs(MODEL_DIR, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                joblib.dump(le, str(le_path))
                joblib.dump(model, str(MODEL_DIR / "model.pkl"))

                acc = model.score(X_test, y_test)
                return model, config, le, f"trained RF with {acc:.0%} accuracy"

    return None, None, None, "error"


def show_result(prediction, warning, chroma_mean, y, sr):
    if warning:
        if "Single note" in warning:
            st.warning(warning)
        else:
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
    model, config, le, model_status = load_model()

    if model_status == "error":
        st.error("No trained model or dataset found.")
    elif model_status == "loaded":
        st.success(f"Model loaded: **{config['model_name'].upper()}**")
    else:
        st.success(f"Model {model_status}")

    st.divider()
    st.caption("Chord Classifier - Major / Minor")

st.title("Chord Classifier")
st.caption("Detect whether a chord is **Major** or **Minor** from microphone or a WAV file.")

if model is None:
    st.error("No model available. Train one first with `python train.py --data_dir data/shords_dataset`.")
    st.stop()

tab_mic, tab_file = st.tabs(["Microphone", "Upload file"])

with tab_mic:
    st.markdown("Record a short clip (about 2 seconds) of a chord on your instrument.")

    try:
        from audio_recorder_streamlit import audio_recorder
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e53935",
            neutral_color="#4fc3f7",
            icon_size="2x",
            pause_threshold=3.0,
            sample_rate=SR,
        )
    except ImportError:
        audio_bytes = None
        st.error(
            "Microphone support requires the `audio_recorder_streamlit` package.\n\n"
            "```\npip install audio-recorder-streamlit\n```"
        )

    if audio_bytes:
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
