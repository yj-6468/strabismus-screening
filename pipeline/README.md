# strabismus-screen

Horizontal strabismus assessment from pediatric cover-test video. From one clinic video plus the declared test type,
fixation distance and correction status, the pipeline outputs:

| task | input | output |
|---|---|---|
| 1. manifest direction | cover-uncover clip | esotropic / exotropic / orthotropic |
| 2. total horizontal deviation | alternate-cover clip | prism diopters (Δ) and direction, or *needs examination* |

**Investigational research software. Not a medical device; not for clinical decisions.**

## Results (patient-grouped 5-fold cross-validation, 774 clips, 109 children)

| | result |
|---|---|
| Task 2, clips graded measurable (235/311, 76%) | MAE 5.82 Δ [4.93, 6.74], ICC 0.796 [0.732, 0.844], 73% within 8 Δ |
| Task 2, every clip (311) | MAE 7.70 Δ [6.59, 8.99], ICC 0.606 [0.478, 0.704] |
| Task 1 (456 clips) | balanced accuracy 0.714 [0.657, 0.770]; direction correct in 95% of detected tropias |

The results write-up and the full research log name individual clips, so they are kept locally and are not part of
this repository.

## How it works

1. **Eye tracking** (`vision.py`, `native_limbus.py`): MediaPipe Face Landmarker localises face and eyes on every
   frame; the limbus (iris edge) is then fitted at sub-pixel precision on the native-resolution frame.
2. **Patient identification** (`track_merge.py`): face-track fragments broken by an occluder crossing the face are
   re-joined into one person before the patient is chosen.
3. **Cover-state detection** (`events.py`): a classifier trained on frame annotations infers, per frame, which eye is
   covered; cover/uncover events follow.
4. **Measurement** (`native_features.py`, `features.py`): eye displacement around each event, converted to
   prism-diopter equivalents (Δ = 100·tan θ).
5. **Prediction** (`native_bundle.py`): gradient-boosted / ensemble heads per task, plus an abstention layer that
   declines magnitude on clips the front end could not measure (face seen on < 50% of frames, no usable eye-movement
   sample, or among the least measurable 20%).

## Layout

```
src/strabismus_screen/   the package (data loading, labels, vision, features, models, CLI)
scripts/final/           build_fold_tables.py, final_numbers.py, reproduce_final.py: produce the reported results
scripts/evaluation/      harness.py (leakage-safe evaluation), common.py (label policy, metadata), run_heads.py (model heads)
configs/                 pipeline settings (stride1.yaml, default.yaml)
tests/                   unit and regression tests
```

## What is not in the repository

`.gitignore` is a whitelist: only `src/`, `scripts/`, `tests/`, `configs/*.yaml`, `pyproject.toml` and this README are tracked. Everything else stays local: clinical videos, the label-policy CSVs
(`configs/*.csv`, which contain clip filenames), the research log, the trained model and caches (`artifacts/`,
rebuilt by `scripts/final/reproduce_final.py`), per-clip predictions (`reports/`) and the MediaPipe asset (`models/`).
The scripts expect this local layout:

```
Strabismus/
├── data/final_dataset/      the 774-clip dataset (videos, chart tables, frame annotations)
└── pipeline/                this repository
    ├── configs/*_v2.csv     label-policy CSVs (copy in data/final_dataset/label configs/)
    ├── models/face_landmarker.task
    └── artifacts/           final_model_v14.joblib, observations_v14_merged/, v14_folds_h0/
```

The dataset is found relative to this folder (`../data/final_dataset`), so the parent folder may be moved or renamed.

## Setup

```
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,native]"
```

Download `face_landmarker.task` (MediaPipe Face Landmarker) into `models/`.

## Usage

```
# run the final model on one video
.venv/bin/python -m strabismus_screen.cli predict VIDEO.mov --test-type Alternate-Cover --distance near --correction cc

# print every reported number from saved predictions (~2 min)
.venv/bin/python scripts/final/final_numbers.py

# refit the final model from scratch in every fold; writes only to --out (~5 min; add --rebuild-folds for ~12 min)
.venv/bin/python scripts/final/reproduce_final.py --out ~/repro

# train the deployable model on all 774 clips
.venv/bin/python -m strabismus_screen.cli --dataset-root ../data/final_dataset --expected-videos 774 \
    train-native --cache-dir artifacts/observations_v14_merged --output artifacts/final_model_v14.joblib \
    --config configs/stride1.yaml
```

## Tests

```
.venv/bin/python -m pytest
```

Tests needing the dataset or a retired experiment script skip themselves when those are absent.
