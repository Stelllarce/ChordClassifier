"""Single source of truth for constants and the model-config schema.

Everything that used to be re-declared in app.py, the notebook, and features.py
(sample rate, clip length, feature/model names, the JSON keys saved next to each
model) lives here so there is exactly one place to change them.
"""

from pathlib import Path

# --- Audio / feature constants -------------------------------------------------
SR = 22050                 # sample rate every clip is resampled to
TARGET_DURATION = 2.0      # seconds; clips are padded/truncated to this length
HOP_LENGTH = 512           # STFT/CQT hop; 2.0s @ 22050 / 512 -> 87 time frames
N_CHROMA = 12              # 12 pitch classes (C, C#, ... B)

# Number of time frames a normalized clip produces (kept fixed so CNN/RNN inputs
# never need padding). ceil(SR * TARGET_DURATION / HOP_LENGTH).
N_FRAMES = int(-(-(SR * TARGET_DURATION) // HOP_LENGTH))  # == 87

# --- Feature types -------------------------------------------------------------
# Aggregated chroma statistics (mean+std over time) -> a flat vector.
FEATURE_CHROMA_VECTOR = "chroma_vector"
# Full 2D chromagram (12 pitch classes x N_FRAMES) treated as an image/sequence.
FEATURE_CHROMAGRAM = "chromagram"

# --- Model types ---------------------------------------------------------------
MODEL_SKLEARN = "sklearn"  # persisted as model.pkl (a StandardScaler pipeline)
MODEL_KERAS = "keras"      # persisted as model.keras

# Which model keys use which feature representation.
SKLEARN_MODELS = ["rf", "svm", "knn", "lr"]
KERAS_VECTOR_MODELS = ["mlp"]
KERAS_SEQUENCE_MODELS = ["cnn", "lstm", "gru"]

# --- Labels --------------------------------------------------------------------
LABELS = ["Major", "Minor"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --- Silence / noise gate ------------------------------------------------------
RMS_SILENCE_THRESHOLD = 0.005  # below this loudness we treat the clip as silence

# --- Paths (relative to the project root, i.e. the ChordClassifier/ folder) -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

AUDIO_EXTENSIONS = [".wav", ".mp3"]
