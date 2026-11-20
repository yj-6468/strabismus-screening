# Strabismus — horizontal strabismus from cover-test video

## Layout

```
Strabismus/
├── README.md                  this file
├── data/
│   ├── final_dataset/         16 GB  the 774 clips (456 cover-uncover, 318 alternate-cover; 109 children) the
│   │                                 final model was trained and evaluated on, with chart tables, frame
│   │                                 annotations and label configs  (was ~/Desktop/CHLA Data)
│   ├── raw_recordings/        94 GB  the ORIGINAL full-visit clinic recordings the clips were cut from, one
│   │                                 folder per patient (205): 385 near/far cover, 220 motility exams (never
│   │                                 used by any model), 56 live-camera captures. Treat as irreplaceable.
│   └── clinical_records/             Data_5.1.26.xlsx — clinic chart export (265 visit rows)
│                                     phase0 video log.xlsx — per-video quality notes (two people in frame,
│                                       face partial/occluded, …)
│                                     early_label_tables/ — label tables from the first phase (May–Jun 2026)
├── pipeline/                  1.8 GB  code, final model, cached eye tracking, results  (was ~/Desktop/STRA)
│   ├── README.md              the code's own front page (install, layout, commands)
│   ├── docs/                  RESEARCH_LOG.md, RESULTS_AND_REPRODUCTION.md, REORGANIZATION.md
│   ├── src/                   the strabismus_screen package
│   ├── artifacts/             final_model_v14.joblib, observations_v14_merged/, v14_folds_h0/
│   ├── scripts/final/         final_numbers.py, reproduce_final.py, build_fold_tables.py
│   ├── scripts/evaluation/    harness.py, common.py, run_heads.py (evaluation harness + model heads)
│   └── configs/ models/ reports/v14/ tests/ .venv/
├── reference/
│   ├── literature/            Esodeviations & Exodeviations; Exotropia (textbook chapters)
│   └── AIStrabMeasure_Code/   collaborator's repository: manifest deviation from 9-gaze photographs
└── archive/
    ├── ixt-sciencefair/       the finished science-fair IXT project (Korean cohort, n = 33) — frozen, unchanged
    └── ixt-sciencefair.requirements.txt   package list to recreate its Python environment
```

## Final results (patient-grouped 5-fold cross-validation)

- **Magnitude** (311 alternate-cover clips): MAE 7.70 Δ, ICC 0.61 answering every clip; **MAE 5.82 Δ
  [4.93, 6.74], ICC 0.80 [0.73, 0.84], 73% within 8 Δ** on the 76% of clips graded measurable (the rest flagged for
  examination).
- **Direction** (456 cover-uncover clips): balanced accuracy 0.714 [0.657, 0.770]; direction correct in 95% of
  detected tropias; sensitivity 87% / specificity 58% at the model's decision point.

## Common commands (run inside `pipeline/`)

```
cd ~/Documents/Strabismus/pipeline
.venv/bin/python scripts/final/final_numbers.py                         # every reported number (~2 min)
.venv/bin/python scripts/final/reproduce_final.py --out ~/repro          # refit from scratch, writes only to ~/repro
.venv/bin/python -m strabismus_screen.cli predict VIDEO.mov \
    --test-type Alternate-Cover --distance near --correction cc          # run the final model on one video
```

The code finds the dataset relative to its own folder (`../data/final_dataset`), so this whole `Strabismus`
folder can be moved or renamed without breaking anything. Investigational only — not for clinical decisions.
