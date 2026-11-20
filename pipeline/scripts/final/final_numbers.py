"""Every number quoted for the final (v14) model, from saved outputs only.

Inputs (defaults = the archived v14 run):
  --data     <Strabismus>/data/final_dataset            chart tables (cohort description)
  --folds    artifacts/v14_folds_h0                    quality columns for the abstention layer
  --oof-dir  reports/v14                               v14h0_patient_job{1,2}_oof.csv
  --before   reports/v8/predictions_778.csv            OPTIONAL: original-video predictions, only for
                                                       the before/after (paired) comparison
Bootstrap: patient-clustered, 4000 draws, seed 21.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT.parent / "data" / "final_dataset")
ap.add_argument("--folds", type=Path, default=ROOT / "artifacts/v14_folds_h0")
ap.add_argument("--oof-dir", type=Path, default=ROOT / "reports/v14")
ap.add_argument("--oof-prefix", default="v14h0_patient_job")
ap.add_argument("--before", type=Path, default=ROOT / "reports/v8/predictions_778.csv")
a = ap.parse_args()
N = "no_manifest_horizontal_deviation"
MEAS = ("face_observation_fraction", "nat_q_right_conf_fraction", "nat_q_left_conf_fraction",
        "all_usable_n", "cover_state_known_fraction")


def icc(x, y):  # ICC(2,1): two-way random, absolute agreement, single measures
    M = np.column_stack([np.asarray(x, float), np.asarray(y, float)]); n, k = M.shape; gm = M.mean()
    MSR = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1); MSC = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    r = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm
    MSE = (r ** 2).sum() / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def ci(df, fn, n=4000, seed=21):
    rng = np.random.default_rng(seed); pats = df.patient.unique()
    idx = {p: np.flatnonzero((df.patient == p).values) for p in pats}; out = []
    for _ in range(n):
        s = df.iloc[np.concatenate([idx[p] for p in rng.choice(pats, len(pats), True)])]
        try:
            v = fn(s)
            if np.isfinite(v): out.append(v)
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5])


def ba(d, classes):
    return np.mean([(d[d.truth == c].pred == c).mean() for c in classes if (d.truth == c).any()])


# ---------------- cohort (chart tables of the final dataset)
ch = pd.concat([pd.read_csv(a.data / "Older data/CHLA v2.csv"),
                pd.read_csv(a.data / "New data/Additional CHLA dataset.csv")], ignore_index=True)
ch["pid"] = ch["Patient ID"].astype(str)
cat = lambda h: "eso" if str(h).strip() in ("ET", "E(T)") else "exo" if str(h).strip() in ("XT", "X(T)") else "ortho"
ch["cat"] = ch["Horizontal type"].map(cat)
pl = {}
for p, g in ch.groupby("pid"):
    s = set(g.cat)
    if {"eso", "exo"} <= s:  # deviations in both directions at different visits -> predominant direction
        pl[p] = g.cat[g.cat != "ortho"].value_counts().idxmax()
    else:
        pl[p] = "eso" if "eso" in s else "exo" if "exo" in s else "ortho"
pl = pd.Series(pl)
def rng_pd(v):
    n = [float(x) for x in re.findall(r"\d+\.?\d*", str(v))]; return (min(n), max(n)) if n else None
ch["pdr"] = ch["Horizontal PD"].map(rng_pd)
sess = ch.dropna(subset=["pdr"]).drop_duplicates(["pid", "Date", "Visual acuity", "Horizontal type", "Horizontal PD"])
print("COHORT")
print(f"  clips {len(ch)}: cover-uncover {(ch['Test type']=='Cover-Uncover').sum()}, alternate-cover {(ch['Test type']=='Alternate-Cover').sum()}; children {ch.pid.nunique()}")
print(f"  children: ET {(pl=='eso').sum()}, XT {(pl=='exo').sum()}, orthotropic {(pl=='ortho').sum()}"
      f"  (mixed-direction children assigned to predominant direction: {sum(1 for p,g in ch.groupby('pid') if {'eso','exo'}<=set(g.cat))})")
for k, nm in (("eso", "ET"), ("exo", "XT")):
    g = sess[sess.cat == k]; print(f"  {nm} range {min(x[0] for x in g.pdr):g}-{max(x[1] for x in g.pdr):g} PD ({len(g)} sessions)")

# ---------------- Job 2: magnitude
o = pd.read_csv(a.oof_dir / f"{a.oof_prefix}2_oof.csv")
o["mid"] = (o.truth_lower + o.truth_upper) / 2; o["err"] = (o.emitted_pd - o["mid"]).abs()
qs = []
for f in range(5):
    t = pd.read_csv(a.folds / f"patient_fold{f}.csv", low_memory=False)
    qs.append(t.loc[t["split"] == "test", ["video_uid", *[c for c in MEAS if c in t.columns]]].assign(qfold=f))
q = pd.concat(qs).drop_duplicates("video_uid").reset_index(drop=True)
X = q[list(MEAS)].to_numpy(float); fq = q.qfold.to_numpy(); meas = np.zeros(len(q))
for f in np.unique(fq):  # nested rank: each fold ranked against the OTHER folds only
    te, tr = fq == f, fq != f; per = np.zeros((te.sum(), X.shape[1]))
    for j in range(X.shape[1]):
        ref = np.sort(X[tr, j][np.isfinite(X[tr, j])]); v = X[te, j]
        r = np.searchsorted(ref, v, side="right") / max(len(ref), 1); r[~np.isfinite(v)] = 0.0; per[:, j] = r
    meas[te] = per.mean(1)
q["measurability"] = meas
m = o.merge(q, on="video_uid", how="left", validate="one_to_one")
gate = (m.face_observation_fraction >= 0.50) & (m.all_usable_n > 0) & (m.cover_state_known_fraction >= 0.10)
tau = m.qfold.map({f: np.quantile(m.measurability[m.qfold != f], 0.20) for f in m.qfold.unique()})
m["report"] = gate & (m.measurability >= tau)
MAE = lambda s: s.err.mean(); ICC = lambda s: icc(s["mid"], s.emitted_pd)
print("\nJOB 2 - total horizontal deviation (alternate-cover clips with a charted PD)")
for nm, d in (("all clips", m), ("reported (abstention)", m[m.report])):
    c1, c2 = ci(d, MAE), ci(d, ICC)
    w = lambda t: 100 * (d.err <= t).mean()
    print(f"  {nm:22s} n={len(d)} ({100*len(d)/len(m):.0f}%)  MAE {MAE(d):.2f} [{c1[0]:.2f}, {c1[1]:.2f}]  "
          f"ICC {ICC(d):.3f} [{c2[0]:.3f}, {c2[1]:.3f}]  within 5/8/10 PD {w(5):.0f}/{w(8):.0f}/{w(10):.0f}%")
print(f"  declined: {(~m.report).sum()} ({(~gate).sum()} by the hard gate, {(gate & ~m.report).sum()} by the measurability threshold)")

# ---------------- Job 1: manifest direction (cover-uncover)
j1 = pd.read_csv(a.oof_dir / f"{a.oof_prefix}1_oof.csv"); C1 = ["eso_tropia", "exo_tropia", N]
b = ba(j1, C1); bc = ci(j1, lambda s: ba(s, C1))
y = (j1.truth != N).values; pt = (j1.pred != N).values; sc = (1 - j1[f"p_{N}"]).values
both = j1[(j1.pred != N) & (j1.truth != N)]
print("\nJOB 1 - manifest direction (cover-uncover)")
print(f"  balanced accuracy {b:.3f} [{bc[0]:.3f}, {bc[1]:.3f}]; recall eso/exo/ortho "
      + " / ".join(f"{(j1[j1.truth==c].pred==c).mean():.2f}" for c in C1))
print(f"  any-tropia detection at the model's decision: sensitivity {pt[y].mean():.2f}, specificity {(~pt[~y]).mean():.2f}")
print(f"  direction correct when a tropia was detected: {(both.pred==both.truth).sum()}/{len(both)} = {100*(both.pred==both.truth).mean():.0f}%")
for sp in (0.80, 0.95):
    thr = np.quantile(sc[~y], sp); print(f"  sensitivity at {int(sp*100)}% specificity (pooled threshold): {(sc[y]>=thr).mean():.2f}")

# ---------------- optional: before/after the clip edits (needs the original predictions)
if a.before.exists():
    old = pd.read_csv(a.before)
    old = old[old.test_type == "Alternate-Cover"].dropna(subset=["job2_pred_pd", "job2_truth_pd_mid"]).copy()
    old["err"] = (old.job2_pred_pd - old.job2_truth_pd_mid).abs(); old["rep"] = old.job2_decision == "report"
    slot = lambda k: k if k in set(m.source_video_key) else k.rsplit(".", 1)[0] + "_fixed.mp4"
    old["slot"] = old.source_video_key.map(slot)
    j = old.merge(m[["source_video_key", "mid", "emitted_pd", "err", "report"]].rename(columns={"source_video_key": "slot"}), on="slot")
    jr = j[j.rep & j.report]
    dm = lambda s: s.err_y.mean() - s.err_x.mean()
    di = lambda s: icc(s.mid, s.emitted_pd) - icc(s.job2_truth_pd_mid, s.job2_pred_pd)
    print(f"\nBEFORE vs AFTER the 7 re-edited clips (paired; {len(j)} clip slots, {len(jr)} reported in both runs)")
    for nm, d in (("all slots", j), ("reported in both", jr)):
        c1, c2 = ci(d, dm), ci(d, di)
        print(f"  {nm:17s} MAE delta {dm(d):+.2f} [{c1[0]:+.2f}, {c1[1]:+.2f}]   ICC delta {di(d):+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}]")
else:
    print("\n(before/after comparison skipped: original predictions file not present)")
