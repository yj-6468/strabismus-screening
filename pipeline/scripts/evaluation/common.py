"""Shared helpers for the evaluation scripts: dataset location, label policy, evaluation metadata.

The metadata (distance, edited flag, pseudo-strabismus flag) is for slicing results only, never a feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

# The final dataset sits next to this pipeline folder: <Strabismus>/data/final_dataset (774 clips).
DATASET_ROOT = ROOT.parent / "data" / "final_dataset"
EXPECTED_VIDEOS = 774
PSEUDO_CSV = ROOT / "configs" / "pseudo_strabismus_v2.csv"
# Byte-verified: a new-named clip is "re-exported" only if its bytes differ from
# the original it replaces.  A filename test is wrong here: one `_SN` clip is a
# byte-identical rename of its original.
REEXPORTED_CSV = ROOT / "configs" / "reexported_clips_v2.csv"
J1_NORMAL = "no_manifest_horizontal_deviation"
# Expected Job 1 distribution under the clinician's policy (phoria-only and
# well-controlled intermittent tropias are orthotropic).  Asserted on load so a
# table built by an older pipeline cannot be scored by mistake.
J1_EXPECTED_COUNTS = {"exo_tropia": 193, "no_manifest_horizontal_deviation": 186, "eso_tropia": 77}


def video_metadata() -> pd.DataFrame:
    from strabismus_screen.data import load_dataset

    dataset = load_dataset(DATASET_ROOT, expected_video_count=EXPECTED_VIDEOS)
    pseudo = pd.read_csv(PSEUDO_CSV).set_index("source_video_key")["kind"].to_dict()
    pairs = pd.read_csv(REEXPORTED_CSV)
    reexported = set(pairs.loc[~pairs.byte_identical_rename, "rel_new"])
    rows = []
    for record in dataset.manifest:
        key = f"{record.patient_id}/{Path(record.relative_path).name}"
        rows.append({"video_uid": record.video_uid, "source_video_key": key, "condition": record.condition,
                     "edited": record.relative_path in reexported, "pseudo": pseudo.get(key, "")})
    meta = pd.DataFrame(rows)
    if int(meta.edited.sum()) != len(reexported):
        raise ValueError("re-exported clip list does not match the manifest")
    missing = set(pseudo) - set(meta.source_video_key)
    if missing:
        raise ValueError(f"pseudo-strabismus keys missing from the manifest: {sorted(missing)}")
    return meta


def load_tables(directory: Path, mode: str, folds: int = 5) -> list[pd.DataFrame]:
    import run_heads as ref

    tables = []
    for fold in range(folds):
        table = ref.add_derived(pd.read_csv(Path(directory) / f"{mode}_fold{fold}.csv", low_memory=False))
        # The labels come from the pipeline (strabismus_screen.labels), never from a
        # rule re-implemented here: Job 1 now depends on the chart control score,
        # which is not a column of the feature table.
        counts = table.loc[table["job1_label"].notna(), "job1_label"].value_counts().to_dict()
        if counts != J1_EXPECTED_COUNTS:
            raise ValueError(
                f"{directory}/{mode}_fold{fold}.csv was built under a different Job 1 policy "
                f"({counts} vs {J1_EXPECTED_COUNTS}); rebuild the fold tables"
            )
        tables.append(table)
    return tables
