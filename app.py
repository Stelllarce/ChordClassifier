import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import io
import os
import sys
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_all_features, normalize_duration, extract_chroma
from src.dataset import load_dataset, split_dataset
from src.models import get_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR = 22050
TARGET_DURATION = 2.0
MODEL_PATH = PROJECT_ROOT / "trained_model.pkl"
DATA_DIRS = [
    PROJECT_ROOT / "data" / "shords_dataset",
    PROJECT_ROOT / "data" / "Audio_Files",
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def is_chord(y: np.ndarray, sr: int,
             chroma_threshold: float = 0.45,
             min_chord_notes: int = 3,
             rms_threshold: float = 0.005) -> tuple[bool, str, np.ndarray]:
    """Decide whether *y* contains a chord, a single note, or silence.

    Returns (is_chord_flag, label, chroma_mean).
    """
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


def process_audio_bytes(audio_bytes: bytes, sr: int = SR):
    """Load audio waveform from raw WAV/audio bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        y, sr_out = librosa.load(tmp.name, sr=sr)
    finally:
        os.unlink(tmp.name)
    return y, sr_out


def classify_audio(y: np.ndarray, sr: int, model):
    """Return (prediction, warning_message, chroma_mean).

    *prediction* is ``None`` when the audio is not a chord.
    """
    y = normalize_duration(y, sr, target_duration=TARGET_DURATION)
    is_chord_flag, audio_type, chroma_mean = is_chord(y, sr)

    if audio_type == "silence":
        return None, "No audio detected — please play a chord.", chroma_mean

    if not is_chord_flag:
        return None, "Single note detected — please play a chord!", chroma_mean

    features = extract_all_features(y, sr)
    prediction = model.predict(features.reshape(1, -1))[0]
    return prediction, None, chroma_mean


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_chroma(chroma_mean: np.ndarray):
    """Bar chart of chroma energy."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    colours = ["#4fc3f7" if v < np.max(chroma_mean) * 0.45 else "#ff7043"
               for v in chroma_mean]
    ax.bar(NOTE_NAMES, chroma_mean, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Energy")
    ax.set_title("Chroma Profile")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_waveform(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4fc3f7")
    ax.set_title("Waveform")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------

@st.cache_resource
def load_or_train_model(model_name: str = "rf"):
    """Load a saved model or train one from the bundled dataset."""
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model, "loaded"

    for data_dir in DATA_DIRS:
        if data_dir.exists() and any(data_dir.iterdir()):
            features, labels = load_dataset(str(data_dir))
            if len(features) > 0:
                X_train, X_test, y_train, y_test = split_dataset(features, labels)
                model = get_model(model_name)
                model.fit(X_train, y_train)
                joblib.dump(model, str(MODEL_PATH))
                acc = model.score(X_test, y_test)
                return model, f"trained — {acc:.0%} accuracy"

    return None, "error"


# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------

def show_result(prediction, warning, chroma_mean, y, sr):
    """Render the classification result block."""
    if warning:
        if "Single note" in warning:
            st.warning(warning, icon="🎵")
        else:
            st.info(warning, icon="🔇")
    else:
        label = str(prediction).capitalize()
        if label.lower() == "major":
            st.success(f"**{label} chord** detected", icon="🎶")
        else:
            st.error(f"**{label} chord** detected", icon="🎶")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_waveform(y, sr))
    with col2:
        st.pyplot(plot_chroma(chroma_mean))


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Chord Classifier", page_icon="🎸", layout="centered")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {min-width:260px; max-width:320px;}
    div[data-testid="stFileUploader"] > div {border: 2px dashed #4fc3f7; border-radius:1rem; padding:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model algorithm",
        options=["rf", "svm", "knn", "lr"],
        format_func=lambda x: {"rf": "Random Forest", "svm": "SVM",
                                "knn": "K-Nearest Neighbours",
                                "lr": "Logistic Regression"}[x],
    )

    model, model_status = load_or_train_model(model_choice)

    if model_status == "error":
        st.error("No trained model or dataset found.")
    elif model_status == "loaded":
        st.success("Model loaded from file")
    else:
        st.success(f"Model {model_status}")

    st.divider()
    st.caption("Chord Classifier · Major / Minor")

# ── Title ──────────────────────────────────────────────────────────────────
st.title("Chord Classifier")
st.caption("Detect whether a chord is **Major** or **Minor** — from microphone or a WAV file.")

if model is None:
    st.error("No model available. Place a `trained_model.pkl` in the project root "
             "or ensure training data is in `data/`.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_mic, tab_file = st.tabs(["🎤  Microphone", "📁  Upload file"])

# ── Microphone tab ─────────────────────────────────────────────────────────
with tab_mic:
    st.markdown("Record a short clip (≈ 2 s) of a chord on your instrument.")

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
        with st.spinner("Analysing…"):
            y, sr = process_audio_bytes(audio_bytes, SR)
            prediction, warning, chroma_mean = classify_audio(y, sr, model)
        show_result(prediction, warning, chroma_mean, y, sr)

# ── File-upload tab ────────────────────────────────────────────────────────
with tab_file:
    st.markdown("Drag-and-drop or browse for a `.wav` file.")

    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=["wav"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Analysing…"):
            y, sr = process_audio_bytes(audio_bytes, SR)
            prediction, warning, chroma_mean = classify_audio(y, sr, model)
        show_result(prediction, warning, chroma_mean, y, sr)
