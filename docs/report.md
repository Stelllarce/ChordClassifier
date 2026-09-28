# ChordClassifier — Changelog & Explanations

This document explains, in plain language, every round of changes made to the
project: **what changed, why it matters, and what to expect**. It's written to be
read without remembering the technical details. Newest phase at the bottom.

---

## Background: what the project does

The app listens to a short audio clip of a chord and decides whether it is
**major** (happy-sounding) or **minor** (sad-sounding). The only real difference
between the two is one note inside the chord — the *third*. Everything below is
about making the computer focus on that difference and nothing else.

The plan has five phases. This report grows one section per phase.

---

## Phase 1 — Cleaning up the foundations

**The problem.** The code had grown messy in ways that make bugs easy and
maintenance hard:

- The same basic settings (audio sample rate, clip length) were written out in
  three different files. Change one, forget another, and things silently break.
- The instructions for *how to save* a trained model lived in the training
  notebook, while the instructions for *how to load* it lived in the app — two
  copies of the same rulebook that could drift apart.
- One helper file (`dataset.py`) prepared the audio in a *different* way than the
  code that was actually used for training, so it was misleading dead weight.
- The README described files that don't exist.

**What changed.**

- Added **`src/config.py`** — one file that holds every shared setting and the
  definition of a model's "config" (the small description saved next to each
  model). Everything now reads from here, so there's a single place to change.
- Added **`src/persistence.py`** — one file that knows how to save, load, and
  discover models. The save/load rulebook now exists exactly once.
- Rewrote **`src/dataset.py`** so it prepares audio with the *same* functions the
  app uses at prediction time. Training and the live app can no longer disagree.
- Pinned the **requirements**, rewrote the **README**, and updated the project
  guide so they match reality.

**Why it matters.** None of this changes what the user sees yet, but it removes
whole categories of "works in training, breaks in the app" bugs and makes the
later phases much safer to build on.

**A promising early result.** While reorganising the audio processing, I switched
the core feature to a musically-aware one (CQT chroma) and tested the "rotation
trick" described in Phase 2. On a quick check, the classic models jumped from
their old scores to roughly **0.92–0.93** (out of 1.0) — a large improvement —
which confirms the direction for the next phase. Full retraining of every model
happens in Phase 3.

---

## Phase 2 — Better features and honest data analysis

**The problem.** The old features mixed two things together: *chroma* (which pitch
notes are present — directly useful) and *MFCC* (which describes the instrument's
tone — mostly irrelevant to major-vs-minor). Because the two datasets are
different instruments, the MFCC part let models cheat by guessing *which dataset*
a clip came from instead of learning the actual chord. On top of that, chroma is
"absolute": a C-major chord and a D-major chord look completely different, so the
model had to learn every key from scratch with limited examples.

**What changed.**

- **Switched to CQT chroma** as the one and only model input. CQT lines up with
  musical pitch, giving a cleaner picture of the notes in the chord. MFCC is now
  kept only for exploration, never fed to a model — closing the "guess the
  dataset" loophole.
- **Added the rotation trick (pitch augmentation).** Since a chord shifted to
  another key is still major/minor, we make 11 extra shifted copies of every clip.
  This teaches models to ignore the root note and multiplies the training data
  about 12×, for free. (Important detail: copies are only added to the *training*
  half, after splitting off the test set, so the test remains a fair exam.)
- **Added noisy-data handling.** A loudness gate rejects silent clips, and a
  tuning check measures how far a recording drifts from the standard 440 Hz.
- **Added a clean analysis notebook** (`notebooks/eda.ipynb`) that shows: the
  class balance across both datasets, the silence/tuning checks, the average
  chroma per class (you can literally see the "third" that separates major from
  minor), and a PCA plot where the two classes already pull apart.

**What the analysis found.** 1361 clips total, roughly balanced (752 major / 609
minor). None were silent. Tuning is close to standard, with only 4 clips drifting
noticeably. The major/minor difference shows up on exactly the pitch classes music
theory predicts.

**Why it matters.** These are the changes that actually make the models better and
more trustworthy — measured improvements come in Phase 3. The older exploratory
notebooks are kept for history, but `eda.ipynb` is now the canonical, up-to-date
analysis.

---

## Phase 3 — Rebuilding and retraining every model

**The problem.** The models were a mixed bag. The best (CNN) was decent, but one
of the required models (the RNN) was **broken** — it scored 0.39, worse than
flipping a coin. There was also no clean, repeatable way to retrain: all the logic
lived inside a notebook full of Google-Colab-specific paths.

**What changed.**

- **Added one training program** (`src/train.py`) that trains the whole set of
  models and saves them. You can run it with a single command, and the Colab
  notebook now just calls it. No more logic buried in notebook cells.
- **Rewrote the Colab notebook** (`notebooks/model_training.ipynb`) to only do two
  things: download the data and call the training program. Much easier to follow.
- **Fixed the recurrent model.** The old RNN was fed the wrong kind of input. It
  now reads the chord as a short sequence of chroma frames, and — as the
  assignment asks — we train **both an LSTM and a GRU** version and compare them.
- **Switched the CNN** to read the chromagram (the pitch-vs-time picture) instead
  of a generic spectrogram, which is more directly relevant to the task.
- **Retrained everything** on the new features with the rotation trick and saved
  fresh scores next to each model.

**Results (test F1, higher is better, out of 1.0):**

| Model | Before | After |
|-------|:------:|:-----:|
| GRU (recurrent)  | 0.39 (old RNN) | **0.967** |
| LSTM (recurrent) | 0.39 (old RNN) | **0.967** |
| CNN              | 0.866 | **0.963** |
| Random Forest    | 0.806 | **0.930** |
| SVM              | 0.775 | **0.927** |
| KNN              | 0.748 | **0.927** |
| MLP (neural net) | 0.634 | **0.876** |
| Logistic Regression | 0.625 | 0.520 |

**About Logistic Regression.** It's the only model that went *down*, and that's an
honest, expected result worth keeping. Logistic Regression can only draw a
*straight-line* boundary, but "major vs minor" is a relationship between notes, not
something a straight line can separate. Its old 0.625 was actually inflated: the
old features secretly leaked the instrument/dataset, and the linear model latched
onto that shortcut instead of the chord. Once we removed the shortcut (Phase 2),
its true ability showed — roughly a coin flip. We keep it as the "simple baseline"
that demonstrates *why* the smarter models are needed.

**Bottom line.** Every serious model now scores in the **0.88–0.97** range, the
broken model is fixed and is now the best, and retraining is a one-command job.

---

## Phase 4 — A cleaner app

**The problem.** The app worked but was cluttered: hand-written styling hacks, its
own private copy of the model-loading rules, two plots, and no sense of how
*confident* a prediction was.

**What changed.**

- **Reuses the shared code.** The app now loads models through the same
  `persistence` helper the training uses — no duplicated rules that can drift.
- **Shows a confidence score** next to each prediction (e.g. "Minor chord · 99%
  confident"), so you can tell a sure answer from a shaky one.
- **One meaningful picture instead of two.** It draws the chord's chroma and marks
  the **root note** and the **third** — the exact note that makes the chord major
  or minor. The visualization now *explains* the answer.
- **Removed the styling hacks** and simplified the layout to a model picker plus
  two tabs (microphone / file upload).

**Why it matters.** The app is simpler to read and maintain, and it now teaches the
user *why* a chord is major or minor rather than just labelling it.

---

## Phase 5 — Wrap-up

### What the whole project looks like now

- **One place for settings** (`src/config.py`) and **one place for saving/loading
  models** (`src/persistence.py`) — no more duplicated rules that silently drift.
- **Features that fit the task**: musically-aware CQT chroma, no instrument
  leakage, plus the rotation trick that teaches models to ignore the key.
- **A full, working model line-up** scoring 0.88–0.97, including the previously
  broken recurrent model, plus the linear baseline kept for teaching value.
- **A clean analysis notebook**, a **one-command training program**, and a
  **simpler app** that explains its answers.
- **This report**, explaining every step in plain language.

### Where things stand against the assignment

| Assignment step | Status |
|---|---|
| 1. Audio concepts (STFT / CQT / chroma) | Used CQT chroma; explained in README + report |
| 2. Preprocessing & feature extraction | `src/features.py` (chroma vector + chromagram) |
| 3. Class-balance statistics | `notebooks/eda.ipynb` |
| 4. Noisy data (silence, tuning ≠ 440 Hz) | Silence gate + tuning estimate, analysed in EDA |
| 5. Visual analysis (librosa/matplotlib/seaborn) | `notebooks/eda.ipynb` |
| 6. PCA of chroma clusters | `notebooks/eda.ipynb` |
| 7. Predict next chord from context (optional) | Not done — out of scope for a single-chord classifier |
| 8. scikit-learn classifiers (RF/SVM…) | RF/SVM/KNN/LR, best 0.93 |
| 9. 2D CNN on chromagram | CNN 0.963 |
| 10. RNN: LSTM vs GRU | Both trained, 0.967 each |
| 11. Streamlit UI | `app.py` |
| 12. Presentation | Can be assembled from this report (not built) |

### How to retrain in Colab (no technical memory required)

1. Open `notebooks/model_training.ipynb` in Google Colab.
2. Top menu → **Runtime → Change runtime type → GPU** (makes the neural nets fast).
3. Make sure the `ChordClassifier` folder is available in Colab (upload it or clone
   the repo), and set `PROJECT_DIR` in the first cell if needed.
4. **Run all cells** (Runtime → Run all). The notebook will:
   - download the two Kaggle datasets automatically,
   - train all eight models and print a comparison table,
   - show the LSTM-vs-GRU numbers.
5. Run the last cell to zip and download the `models/` folder, then put that folder
   back into the project. The app will automatically use the new models.

That's it — you don't need to touch any of the code to retrain.

**Note on speed:** training on your own machine is slow mainly because reading and
converting ~1400 audio files takes a few minutes, and the neural nets prefer a GPU.
That's exactly why training lives in Colab (free GPU) while your computer only runs
the finished app. The classical models (RF/SVM/KNN) are fast enough to train
anywhere.

