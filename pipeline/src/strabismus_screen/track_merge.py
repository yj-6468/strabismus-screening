"""Patient-identity track merge, applied at training and prediction time.

A paddle or hand crossing the patient's face splits one person into several face tracks,
and ``vision.select_patient_track`` then keeps only one fragment.  This module re-joins
fragments into identities across occluder gaps (position, scale, interocular distance and
appearance continuity; tracks that overlap in time never merge), then applies the original
selection score to the merged identities.

All constants are geometric or photometric; no clinical label is read here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- re-identification constants (geometric; no fit, no label) ---------------
MAX_GAP_FRAMES = 150        # an occluder crossing or a turned head can blank the detector for ~5 s
MAX_CENTRE_JUMP = 0.90      # centre displacement across the gap, in units of the mean face width
SCALE_LO, SCALE_HI = 0.62, 1.60   # face-width ratio across the gap
IOD_LO, IOD_HI = 0.60, 1.67       # interocular-px ratio across the gap (independent scale witness)
APPEARANCE_TOL = 42.0       # max L1 distance over the 6 gray/saturation medians (0-255 units)
ENDPOINT_FRAMES = 5         # frames averaged at each end of a fragment to get its endpoint state
DRIFT_PER_FRAME = 0.004     # extra centre tolerance per frame of gap, in face widths

APPEARANCE_COLS = (
    "right_cheek_gray_q50", "left_cheek_gray_q50",
    "right_cheek_saturation_mean", "left_cheek_saturation_mean",
    "right_eye_gray_q50", "left_eye_gray_q50",
)


# ----------------------------------------------------------------------------- shipped score
def shipped_score(group: pd.DataFrame, sampled_frames: int) -> float:
    """``vision.select_patient_track``'s score, verbatim, for one group of rows.

    Kept as a copy so it can be applied to merged identities.
    """
    right = group["right_cover_proxy"].to_numpy(dtype=float)
    left = group["left_cover_proxy"].to_numpy(dtype=float)
    difference = np.nan_to_num(right - left)
    if len(difference):
        p90 = float(np.percentile(np.abs(difference), 90))
        activity = float(np.std(difference))
        changes = float(np.mean(np.abs(np.diff(difference)) > 0.18)) if len(difference) > 1 else 0.0
    else:
        p90 = activity = changes = 0.0
    coverage = min(1.0, len(group) / max(sampled_frames, 1))
    size = float(np.nanmedian(group["face_area"]))
    dx = group["face_center_x"].to_numpy(dtype=float) - 0.5
    dy = group["face_center_y"].to_numpy(dtype=float) - 0.5
    centrality = float(1.0 - np.clip(np.nanmedian(np.hypot(dx, dy)) / 0.71, 0.0, 1.0))
    return float(
        3.2 * activity
        + 2.0 * p90
        + 1.4 * changes
        + 0.8 * coverage
        + 1.3 * np.sqrt(max(size, 0.0))
        + 0.2 * centrality
    )


# ----------------------------------------------------------------------------- fragments
@dataclass
class Fragment:
    track_id: int
    frames: np.ndarray                 # sorted unique frame indices
    f0: int
    f1: int
    head_c: tuple[float, float]        # centre at the START of the fragment
    tail_c: tuple[float, float]        # centre at the END
    head_w: float
    tail_w: float
    head_iod: float
    tail_iod: float
    head_app: np.ndarray
    tail_app: np.ndarray


def _endpoint(sub: pd.DataFrame, head: bool) -> dict:
    part = sub.head(ENDPOINT_FRAMES) if head else sub.tail(ENDPOINT_FRAMES)
    app = np.array([pd.to_numeric(part[c], errors="coerce").median() if c in part else np.nan
                    for c in APPEARANCE_COLS], dtype=float)
    return {
        "c": (float(part["face_center_x"].median()), float(part["face_center_y"].median())),
        "w": float(part["face_width"].median()),
        "iod": float(pd.to_numeric(part["interocular_px"], errors="coerce").median()),
        "app": app,
    }


def fragments(obs: pd.DataFrame) -> list[Fragment]:
    out: list[Fragment] = []
    for tid, sub in obs.groupby("track_id", sort=True):
        sub = sub.sort_values("frame")
        h, t = _endpoint(sub, True), _endpoint(sub, False)
        fr = np.unique(sub["frame"].to_numpy(dtype=int))
        out.append(Fragment(int(tid), fr, int(fr.min()), int(fr.max()),
                            h["c"], t["c"], h["w"], t["w"], h["iod"], t["iod"], h["app"], t["app"]))
    out.sort(key=lambda f: (f.f0, f.track_id))
    return out


# ----------------------------------------------------------------------------- linking
def _appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return 0.0          # not enough photometry to judge: do not veto on it
    return float(np.mean(np.abs(a[ok] - b[ok])))


def _ratio_ok(a: float, b: float, lo: float, hi: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return True          # missing witness: neither confirms nor vetoes
    return lo <= (a / b) <= hi


def link_cost(tail: Fragment, head: Fragment) -> float | None:
    """Cost of continuing identity ending at ``tail`` with ``head``; None = not the same person."""
    gap = head.f0 - tail.f1
    if gap <= 0:
        return None                                   # overlapping or out of order
    if gap > MAX_GAP_FRAMES:
        return None
    mw = np.nanmean([tail.tail_w, head.head_w])
    if not np.isfinite(mw) or mw <= 0:
        return None
    dist = float(np.hypot(head.head_c[0] - tail.tail_c[0], head.head_c[1] - tail.tail_c[1]))
    tol = (MAX_CENTRE_JUMP + DRIFT_PER_FRAME * gap) * mw
    if dist > tol:
        return None
    if not _ratio_ok(head.head_w, tail.tail_w, SCALE_LO, SCALE_HI):
        return None
    if not _ratio_ok(head.head_iod, tail.tail_iod, IOD_LO, IOD_HI):
        return None
    app = _appearance_distance(head.head_app, tail.tail_app)
    if app > APPEARANCE_TOL:
        return None
    return dist / mw + 0.01 * app + 0.001 * gap


@dataclass
class Identity:
    ident: int
    members: list[int] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)
    tail: Fragment | None = None


def merge(obs: pd.DataFrame) -> tuple[dict[int, int], list[dict]]:
    """track_id -> identity_id, plus one record per accepted link (for auditing)."""
    frags = fragments(obs)
    identities: dict[int, Identity] = {}
    assign: dict[int, int] = {}
    links: list[dict] = []
    for frag in frags:
        best, best_cost = None, np.inf
        for ident, node in identities.items():
            if node.tail is None:
                continue
            # HARD CONSTRAINT: two tracks alive at the same time are two different people.
            # Checked against the identity's WHOLE frame set, not just its last fragment.
            if node.frames & set(frag.frames.tolist()):
                continue
            cost = link_cost(node.tail, frag)
            if cost is not None and cost < best_cost:
                best, best_cost = ident, cost
        if best is None:
            ident = frag.track_id
            identities[ident] = Identity(ident, [frag.track_id], set(frag.frames.tolist()), frag)
            assign[frag.track_id] = ident
        else:
            node = identities[best]
            links.append({"identity": best, "from_track": node.tail.track_id, "to_track": frag.track_id,
                          "gap": int(frag.f0 - node.tail.f1), "cost": float(best_cost),
                          "overlap_frames": len(node.frames & set(frag.frames.tolist()))})
            node.members.append(frag.track_id)
            node.frames |= set(frag.frames.tolist())
            node.tail = frag
            assign[frag.track_id] = best
    return assign, links


def elect(obs: pd.DataFrame, assign: dict[int, int], sampled_frames: int) -> int | None:
    """The SHIPPED score, unchanged weights, applied to merged identities."""
    ident = obs["track_id"].map(assign)
    scores = [(shipped_score(obs[ident == i], sampled_frames), int(i)) for i in sorted(set(assign.values()))]
    return max(scores)[1] if scores else None


def apply(obs: pd.DataFrame, sampled_frames: int) -> tuple[pd.Series, dict]:
    """Return (new is_patient_track, diagnostics) for one video's cached observations."""
    if obs.empty:
        return pd.Series(dtype=bool), {"n_tracks": 0, "n_identities": 0}
    assign, links = merge(obs)
    chosen = elect(obs, assign, sampled_frames)
    ident = obs["track_id"].map(assign)
    flag = ident.eq(chosen)
    shipped = obs["is_patient_track"].astype(bool)
    found = int(obs["frame"].nunique())
    diag = {
        "n_tracks": int(obs["track_id"].nunique()),
        "n_identities": len(set(assign.values())),
        "n_links": len(links),
        "bad_overlap_links": int(sum(l["overlap_frames"] > 0 for l in links)),
        "max_link_gap": int(max([l["gap"] for l in links], default=0)),
        "found": found,
        "shipped_kept": int(obs.loc[shipped, "frame"].nunique()),
        "kept": int(obs.loc[flag, "frame"].nunique()),
        "shipped_track": int(obs.loc[shipped, "track_id"].iloc[0]) if shipped.any() else -1,
        "elected_identity": -1 if chosen is None else int(chosen),
        "elected_members": sorted(int(t) for t, i in assign.items() if i == chosen),
        "changed": bool((flag != shipped).any()),
    }
    diag["contains_shipped_track"] = diag["shipped_track"] in diag["elected_members"]
    return flag, diag


def apply_to_result(result):
    """Return a VisionResult whose patient flag and patient_track_id come from the merged identity.

    Only ``observations['is_patient_track']`` and ``metadata.patient_track_id`` change.
    """
    from dataclasses import replace as _replace
    from .vision import VisionResult
    obs = result.observations
    if obs.empty:
        return result
    flag, diag = apply(obs.sort_values("frame"), int(result.metadata.sampled_frames))
    new_obs = obs.copy()
    new_obs["is_patient_track"] = flag.reindex(obs.index).to_numpy(dtype=bool)
    elected = None if diag["elected_identity"] < 0 else int(diag["elected_identity"])
    return VisionResult(metadata=_replace(result.metadata, patient_track_id=elected), observations=new_obs)
