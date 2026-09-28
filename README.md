# ChordClassifier

Classifies a musical chord as **major** or **minor** from a short audio clip, and
serves the prediction through a Streamlit app.

## How it works

Audio is resampled to 22050 Hz, trimmed/padded to 2 seconds, and turned into
**chroma features** (energy per pitch class C..B) using the Constant-Q Transform.
Major vs. minor is decided by one interval — the third — which is exactly what
chroma captures. Two feature representations are used:

- **chroma vector** — mean + std of chroma over time (24 numbers); used by the
  classical models and the MLP.
- **chromagram** — the full 12 × 87 pitch-class-over-time image; used by the CNN
  and the recurrent (LSTM/GRU) models.

**Pitch-rotation augmentation** is the key trick: a chord shifted to another key
is still major/minor, so rolling the pitch axis creates 11 extra copies per clip.
This makes the models ignore the root note and roughly multiplies the data ×12.

## Project layout

```
src/
  config.py       constants + the model-config schema (single source of truth)
  features.py     audio -> chroma features, augmentation, silence/tuning checks
  dataset.py      walk data/<label>/ folders into feature arrays
  models.py       sklearn factory + Keras builders (mlp/cnn/lstm/gru)
  persistence.py  save/load models + config; discover trained models
  utils.py        plotting helpers for the notebooks
app.py            Streamlit app (microphone or file upload -> prediction)
models/<name>/    a trained model: config.json + label_encoder.pkl + model.*
notebooks/        EDA + the Colab training notebook
docs/report.md    plain-language changelog of what changed and why
```

## Run the app (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app auto-discovers every trained model under `models/`.

## Training

Training is done in **Google Colab** (free GPU) via
`notebooks/model_training.ipynb`. It downloads the datasets with `kagglehub`,
trains the whole model zoo using the shared code in `src/`, and writes each model
back into `models/<name>/`. See `docs/report.md` for a step-by-step, non-technical
walkthrough.

## Datasets

- [major-vs-minor-guitar-chords](https://www.kaggle.com/datasets/mehanat96/major-vs-minor-guitar-chords)
- [musical-instrument-chord-classification](https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification)

`data/` is git-ignored; the notebook downloads it on demand.
