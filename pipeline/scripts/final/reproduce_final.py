"""Reproduce the final (v14) results from <Strabismus>/data/final_dataset. Writes ONLY to --out.

  .venv/bin/python scripts/final/reproduce_final.py --out /path/to/new_folder [--rebuild-folds]

1. (--rebuild-folds) rebuild the 5 fold tables from the dataset and the cached eye tracking
   (artifacts/observations_v14_merged), ~7 min.  Otherwise artifacts/v14_folds_h0 is used.
2. Refit the direction, magnitude and four-class heads in every fold, ~5 min.
3. Print every reported number (scripts/final/final_numbers.py).
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data" / "final_dataset"   # <Strabismus>/data/final_dataset
PY = str(ROOT / ".venv/bin/python")
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rebuild-folds", action="store_true")
ap.add_argument("--data", type=Path, default=DATA)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
env = {"PYTHONPATH": ":".join(str(ROOT / p) for p in ("src", "scripts/evaluation"))}
import os; env = {**os.environ, **env}

folds = ROOT / "artifacts/v14_folds_h0"
if a.rebuild_folds:
    folds = a.out / "folds"
    subprocess.run([PY, str(ROOT / "scripts/final/build_fold_tables.py"), "--mode", "patient", "--folds", "5",
                    "--group-recordings", "--stratify-distance", "--dataset-root", str(a.data),
                    "--cache-dir", str(ROOT / "artifacts/observations_v14_merged"), "--out", str(folds)],
                   check=True, cwd=ROOT, env=env)

refit = f"""
import json, sys
from pathlib import Path
import strabismus_screen.data as D; D.EXPECTED_VIDEO_COUNT_V2 = 774
import common
common.DATASET_ROOT = Path({str(a.data)!r})
common.J1_EXPECTED_COUNTS = {{"exo_tropia": 193, "no_manifest_horizontal_deviation": 186, "eso_tropia": 77}}
import harness as H
from threadpoolctl import threadpool_limits
H.MULTIPLICITY_LOG = Path({str(a.out / "multiplicity_log.jsonl")!r})
H.PARTITIONS["final"] = Path({str(folds)!r})
for job in (2, 1, 3):
    fp, view = H.frozen_anchor(job)
    with threadpool_limits(limits=4):
        oof, _ = H.predict_folds(job, "patient", "final", fp, view, name=f"reproduce_final job{{job}}")
    oof.to_csv(Path({str(a.out)!r}) / f"v14h0_patient_job{{job}}_oof.csv", index=False)
    print(f"job{{job}} refit done", flush=True)
"""
subprocess.run([PY, "-c", refit], check=True, cwd=ROOT, env=env)
subprocess.run([PY, str(ROOT / "scripts/final/final_numbers.py"), "--data", str(a.data),
                "--folds", str(folds), "--oof-dir", str(a.out)], check=True, cwd=ROOT, env=env)
