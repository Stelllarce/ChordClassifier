# ChordClassifier

Classifies major and minor musical chords from audio files using machine learning.

## Overview

Audio files are processed to extract chroma and MFCC features using `librosa`. Several classifiers (Random Forest, SVM, KNN, Logistic Regression) are trained and evaluated on these features.

## Project Structure

```
├── data/                  # Audio files organized by chord class (e.g., data/Am/, data/C/)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb  # Feature inspection
│   └── 03_training.ipynb  # Model training and evaluation
├── src/
│   ├── features.py        # Audio feature extraction
│   ├── dataset.py         # Data loading and splitting
│   ├── models.py          # Model definitions
│   └── utils.py           # Plotting and metrics helpers
├── train.py               # Standalone training script
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place audio files in `data/` with one subdirectory per chord class, then:

**Notebooks:** Run notebooks in order (01 → 02 → 03) for analysis and interactive training.

**Script:** `python train.py --data_dir data/ --model rf`

