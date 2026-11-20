"""Event-aligned features from native-resolution limbus measurements (cache v4).

Inputs are the per-frame patient observations produced by
:func:`vision.extract_video_observations` with ``native_limbus=True`` and a
cover timeline (inferred at inference; trusted only for diagnostics).  No
annotation is consumed here.

Coordinates
-----------
``corner`` : nasal-positive limbus position relative to the same-frame canthi
             midpoint, in native pixels (eye-width units times eye width).
``rigid``  : limbus position after a per-frame similarity registration of a
             rigid landmark subset (nose bridge, tragions, chin, forehead,
             cheeks) to the clip's median configuration, nasal-positive.
Both are converted to *uncalibrated prism-diopter equivalents* with the
limbus-translation chain (11.7 mm iris, 0.105 mm per prism diopter) using the
clip's own median limbus radius.  The learned heads calibrate the scale.

Event measurements
------------------
For every inferred uncover event of eye E: pre-cover fixation plateau, first
visible (deviated) position, change-point step amplitude and latency after the
dead zone, late plateau, drift slope, cover duration, head-motion quality.
For every inferred cover event of eye E: the fellow eye's cover-on step
(manifest evidence) measured the same way.  Signs follow the package
convention: positive = eso (temporal refixation of the observed eye).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

IRIS_MM = 11.7
MM_PER_PD = 0.105
RIGID_LANDMARKS = (6, 197, 195, 5, 4, 168, 234, 454, 10, 152, 50, 280)
CONF_GATE = 0.30
EVENT_KINDS = ("unc", "cov")


# ----------------------------------------------------------------------------- helpers
def _finite(a: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(a), dtype=float)
    return x[np.isfinite(x)]


def _median(a: Iterable[float]) -> float:
    x = _finite(a)
    return float(np.median(x)) if len(x) else float("nan")


def _summary(values: list[float], weights: list[float], prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    names = ("n", "median", "mean", "wmean", "dominant", "absmax", "abs_p75", "abs_median", "iqr", "sign_consistency", "positive_fraction")
    if not len(x):
        return {f"{prefix}_{k}": (0.0 if k == "n" else float("nan")) for k in names}
    ax = np.abs(x)
    nontrivial = ax > 0.5
    pos = float(np.mean(x[nontrivial] > 0)) if np.any(nontrivial) else 0.5
    q25, q75 = np.percentile(x, (25, 75))
    return {
        f"{prefix}_n": float(len(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_wmean": float(np.average(x, weights=w)),
        f"{prefix}_dominant": float(x[int(np.argmax(ax))]),
        f"{prefix}_absmax": float(np.max(ax)),
        f"{prefix}_abs_p75": float(np.percentile(ax, 75)),
        f"{prefix}_abs_median": float(np.median(ax)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_sign_consistency": float(max(pos, 1.0 - pos)),
        f"{prefix}_positive_fraction": pos,
    }


def _step_fit(t: np.ndarray, y: np.ndarray, t_min: float, t_max: float) -> tuple[float, float, float, float]:
    """Best single-step (two-plateau) fit of y(t).

    Returns (amplitude = plateau_after - plateau_before, step_time, residual_rms,
    explained_fraction).  The step is searched between t_min and t_max.
    """
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(t)
    t, y = t[order], y[order]
    total_ss = float(np.sum((y - np.mean(y)) ** 2)) + 1e-9
    best = (np.inf, float("nan"), float("nan"))
    for i in range(2, len(y) - 2):
        ts = 0.5 * (t[i - 1] + t[i])
        if ts < t_min or ts > t_max:
            continue
        a, b = np.median(y[:i]), np.median(y[i:])
        ss = float(np.sum((y[:i] - a) ** 2) + np.sum((y[i:] - b) ** 2))
        if ss < best[0]:
            best = (ss, ts, b - a)
    if not np.isfinite(best[1]):
        return float("nan"), float("nan"), float("nan"), float("nan")
    rms = math.sqrt(best[0] / len(y))
    return float(best[2]), float(best[1]), float(rms), float(1.0 - best[0] / total_ss)


def _procrustes(points: np.ndarray, template: np.ndarray):
    mp_, mt = points.mean(0), template.mean(0)
    P0, T0 = points - mp_, template - mt
    denom = float((P0 ** 2).sum())
    if denom < 1e-9:
        return None
    s = math.sqrt(float((T0 ** 2).sum()) / denom)
    U, _, Vt = np.linalg.svd(P0.T @ T0)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return s, R, mp_, mt


# ----------------------------------------------------------------------------- per-frame table
@dataclass
class NativeTrack:
    frames: np.ndarray
    fps: float
    px_per_pd: float
    eye_width_px: float
    right_corner: np.ndarray   # nasal-positive px
    left_corner: np.ndarray
    right_rigid: np.ndarray    # nasal-positive px in rigid frame
    left_rigid: np.ndarray
    right_conf: np.ndarray
    left_conf: np.ndarray
    right_covered: np.ndarray  # timeline state aligned to frames (nan unknown)
    left_covered: np.ndarray
    head_motion: np.ndarray    # rigid-frame translation magnitude per frame (px)
    yaw: np.ndarray

    def pos(self, eye: str, frame_: str) -> np.ndarray:
        return getattr(self, f"{eye}_{frame_}")


def build_track(observations: pd.DataFrame, timeline: pd.DataFrame, *, fps: float) -> NativeTrack | None:
    rows = observations
    if "is_patient_track" in rows:
        sel = rows[rows["is_patient_track"].astype(bool)]
        if not sel.empty:
            rows = sel
    if rows.empty or "right_limbus_x_px" not in rows:
        return None
    rows = rows.sort_values("frame").drop_duplicates("frame", keep="first").reset_index(drop=True)
    frames = rows["frame"].to_numpy(int)
    n = len(rows)

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(rows.get(name), errors="coerce").to_numpy(float) if name in rows else np.full(n, np.nan)

    radii = np.r_[col("right_limbus_r_px"), col("left_limbus_r_px")]
    conf_all = np.r_[col("right_limbus_conf"), col("left_limbus_conf")]
    good_r = radii[np.isfinite(radii) & (conf_all >= CONF_GATE)]
    r_med = float(np.median(good_r)) if len(good_r) >= 5 else float(np.nanmedian(np.r_[col("right_iris_diameter_px"), col("left_iris_diameter_px")]) / 2.0)
    if not np.isfinite(r_med) or r_med < 4:
        return None
    px_per_pd = (2.0 * r_med / IRIS_MM) * MM_PER_PD
    eye_width = float(np.nanmedian(np.r_[col("right_eye_width_px"), col("left_eye_width_px")]))

    # rigid registration
    rigid_xy = np.stack([np.c_[col(f"lm{i}_x_px"), col(f"lm{i}_y_px")] for i in RIGID_LANDMARKS], axis=1)  # n x 12 x 2
    valid_rigid = np.all(np.isfinite(rigid_xy.reshape(n, -1)), axis=1)
    template = np.median(rigid_xy[valid_rigid], axis=0) if valid_rigid.sum() >= 3 else None
    nose_x = col("nose_x_px")

    out = {}
    head_motion = np.full(n, np.nan)
    for eye in ("right", "left"):
        lx, ly = col(f"{eye}_limbus_x_px"), col(f"{eye}_limbus_y_px")
        tx, ty = col(f"{eye}_temporal_canthus_x_px"), col(f"{eye}_temporal_canthus_y_px")
        nx, ny = col(f"{eye}_nasal_canthus_x_px"), col(f"{eye}_nasal_canthus_y_px")
        mid = np.c_[(tx + nx) / 2, (ty + ny) / 2]
        axis = np.c_[nx - tx, ny - ty]
        norm = np.linalg.norm(axis, axis=1)
        axis_n = axis / np.maximum(norm[:, None], 1e-6)
        corner = np.einsum("ij,ij->i", np.c_[lx, ly] - mid, axis_n)  # nasal positive
        rigid = np.full(n, np.nan)
        nasal_sign = np.where(nose_x > mid[:, 0], 1.0, -1.0)
        if template is not None:
            for i in range(n):
                if not valid_rigid[i] or not np.isfinite(lx[i]):
                    continue
                tr = _procrustes(rigid_xy[i], template)
                if tr is None:
                    continue
                s, R, mp_, mt = tr
                p = s * (R @ (np.array([lx[i], ly[i]]) - mp_)) + mt
                rigid[i] = p[0] * nasal_sign[i]
                head_motion[i] = float(np.linalg.norm(mp_ - mt))
        out[f"{eye}_corner"] = corner
        out[f"{eye}_rigid"] = rigid
        out[f"{eye}_conf"] = col(f"{eye}_limbus_conf")

    # timeline alignment
    state = timeline[["frame", "right_covered", "left_covered"]].drop_duplicates("frame", keep="last").set_index("frame")
    rc = state["right_covered"].reindex(frames).to_numpy(float)
    lc = state["left_covered"].reindex(frames).to_numpy(float)
    return NativeTrack(
        frames=frames, fps=float(fps), px_per_pd=float(px_per_pd), eye_width_px=eye_width,
        right_corner=out["right_corner"], left_corner=out["left_corner"],
        right_rigid=out["right_rigid"], left_rigid=out["left_rigid"],
        right_conf=out["right_conf"], left_conf=out["left_conf"],
        right_covered=rc, left_covered=lc, head_motion=head_motion, yaw=col("head_yaw_deg"),
    )


# ----------------------------------------------------------------------------- event measurement
@dataclass
class EventMeasure:
    kind: str            # "unc" (uncover of observed eye) or "cov" (fellow eye during cover of the other)
    observed_eye: str
    frame: int
    frame_: str          # coordinate frame name
    pre_fix: float       # pre-cover fixation plateau (PD-eq, nasal-positive)
    first_visible: float # first visible position after the event
    step_amp: float      # change-point step amplitude (PD-eq, eso-positive)
    step_latency_s: float
    step_quality: float
    late_plateau: float
    deviation_vs_prefix: float   # first_visible - pre_fix (eso-positive)
    late_vs_first: float         # -(late - first) (eso-positive)
    drift_slope: float           # PD-eq per second over the late window (nasal-positive)
    cover_dur_s: float
    open_dur_s: float
    head_motion_px: float
    n_valid: int
    quality: float


def _window(track: NativeTrack, eye: str, frame_: str, start: float, stop: float, *, require_uncovered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    fr = track.frames
    m = (fr >= start) & (fr <= stop) & (track.pos(eye, "conf") >= CONF_GATE)
    if require_uncovered:
        cov = getattr(track, f"{eye}_covered")
        m &= cov == 0.0
    y = track.pos(eye, frame_)
    m &= np.isfinite(y)
    return fr[m].astype(float), y[m]


def measure_event(track: NativeTrack, kind: str, observed_eye: str, f0: int, *, cover_start: int | None, next_cover: int | None, frame_: str) -> EventMeasure | None:
    fps = track.fps
    s = lambda sec: int(round(sec * fps))  # noqa: E731
    stop_limit = (next_cover - 2) if next_cover is not None else f0 + s(1.5)
    # pre-event fixation plateau of the observed eye
    if kind == "unc":
        pre_start, pre_stop = (cover_start - s(0.6), cover_start - 3) if cover_start is not None else (f0 - s(0.6), f0 - 3)
    else:
        pre_start, pre_stop = f0 - s(0.6), f0 - 2
    _, pre = _window(track, observed_eye, frame_, pre_start, pre_stop)
    pre_fix = _median(pre) if len(pre) >= 3 else float("nan")
    # first visible frames after the event (dead zone: use up to 3 frames within 0..4)
    t_first, y_first = _window(track, observed_eye, frame_, f0, f0 + 4)
    first_visible = _median(y_first[:3]) if len(y_first) >= 1 else float("nan")
    # step fit over [f0, stop]
    t_post, y_post = _window(track, observed_eye, frame_, f0, min(f0 + s(1.2), stop_limit))
    amp, t_step, rms, explained = _step_fit(t_post, y_post, f0 + 1.0, f0 + s(0.6)) if len(y_post) >= 6 else (float("nan"),) * 4
    # late plateau
    _, late = _window(track, observed_eye, frame_, f0 + s(0.4), min(f0 + s(1.0), stop_limit))
    late_plateau = _median(late) if len(late) >= 3 else float("nan")
    # drift slope over the late window
    tl, yl = _window(track, observed_eye, frame_, f0 + s(0.3), min(f0 + s(1.5), stop_limit))
    drift = float(np.polyfit(tl / fps, yl, 1)[0]) if len(yl) >= 6 else float("nan")
    # head motion in the event window
    fr = track.frames
    hm = track.head_motion[(fr >= f0 - s(0.6)) & (fr <= min(f0 + s(1.0), stop_limit))]
    hm = hm[np.isfinite(hm)]
    head_motion = float(np.percentile(hm, 90) - np.percentile(hm, 10)) if len(hm) >= 3 else float("nan")
    n_valid = int(len(y_post))
    if n_valid == 0 and not np.isfinite(first_visible):
        return None
    step_quality = float(explained) if np.isfinite(explained) else 0.0
    quality = float(min(1.0, n_valid / 12.0) * (0.5 + 0.5 * step_quality) * (1.0 / (1.0 + max(head_motion, 0.0) / 6.0) if np.isfinite(head_motion) else 0.7))
    return EventMeasure(
        kind=kind, observed_eye=observed_eye, frame=int(f0), frame_=frame_,
        pre_fix=pre_fix, first_visible=first_visible,
        step_amp=float(-amp) if np.isfinite(amp) else float("nan"),  # nasal-positive step -> eso-positive is the negative
        step_latency_s=float((t_step - f0) / fps) if np.isfinite(t_step) else float("nan"),
        step_quality=step_quality, late_plateau=late_plateau,
        deviation_vs_prefix=float(first_visible - pre_fix) if np.isfinite(first_visible) and np.isfinite(pre_fix) else float("nan"),
        late_vs_first=float(-(late_plateau - first_visible)) if np.isfinite(late_plateau) and np.isfinite(first_visible) else float("nan"),
        drift_slope=drift,
        cover_dur_s=float((f0 - cover_start) / fps) if cover_start is not None else float("nan"),
        open_dur_s=float((next_cover - f0) / fps) if next_cover is not None else float("nan"),
        head_motion_px=head_motion, n_valid=n_valid, quality=quality,
    )


def _events_from_timeline(timeline: pd.DataFrame, eye: str, kind: str) -> np.ndarray:
    column = f"{eye}_gets_{'uncovered' if kind == 'unc' else 'covered'}"
    if column not in timeline:
        return np.zeros(0, dtype=int)
    return timeline.loc[timeline[column].astype(bool), "frame"].to_numpy(int)


def measure_all_events(track: NativeTrack, timeline: pd.DataFrame, *, test_type: str) -> list[EventMeasure]:
    measures: list[EventMeasure] = []
    for eye in ("right", "left"):
        fellow = "left" if eye == "right" else "right"
        uncovers = _events_from_timeline(timeline, eye, "unc")
        covers = _events_from_timeline(timeline, eye, "cov")
        fellow_uncovers = _events_from_timeline(timeline, fellow, "unc")
        for f0 in uncovers:
            prev = covers[covers < f0]
            nxt = covers[covers > f0]
            cover_start = int(prev.max()) if len(prev) else None
            next_cover = int(nxt.min()) if len(nxt) else None
            for frame_ in ("corner", "rigid"):
                m = measure_event(track, "unc", eye, int(f0), cover_start=cover_start, next_cover=next_cover, frame_=frame_)
                if m is not None:
                    measures.append(m)
        for fc in covers:
            # In ACT a cover of this eye is paired with the fellow's uncover; the fellow's motion is then an uncover measurement.
            if test_type == "Alternate-Cover" and len(fellow_uncovers) and np.min(np.abs(fellow_uncovers - fc)) <= 2:
                continue
            fellow_covers = _events_from_timeline(timeline, fellow, "cov")
            nxt = fellow_covers[fellow_covers > fc]
            next_cover = int(nxt.min()) if len(nxt) else None
            for frame_ in ("corner", "rigid"):
                m = measure_event(track, "cov", fellow, int(fc), cover_start=None, next_cover=next_cover, frame_=frame_)
                if m is not None:
                    measures.append(m)
    return measures


# ----------------------------------------------------------------------------- aggregation
def _binocular_features(track: NativeTrack, frame_: str, prefix: str) -> dict[str, float]:
    both = (track.right_covered == 0.0) & (track.left_covered == 0.0) & (track.right_conf >= CONF_GATE) & (track.left_conf >= CONF_GATE)
    r = track.pos("right", frame_)[both] / track.px_per_pd
    l = track.pos("left", frame_)[both] / track.px_per_pd
    ok = np.isfinite(r) & np.isfinite(l)
    r, l = r[ok], l[ok]
    out = {f"{prefix}_n": float(len(r))}
    if len(r) < 3:
        for k in ("vergence_median", "vergence_iqr", "right_median", "left_median", "right_iqr", "left_iqr", "diff_median"):
            out[f"{prefix}_{k}"] = float("nan")
        return out
    v = r + l
    out.update({
        f"{prefix}_vergence_median": float(np.median(v)),
        f"{prefix}_vergence_iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
        f"{prefix}_right_median": float(np.median(r)),
        f"{prefix}_left_median": float(np.median(l)),
        f"{prefix}_right_iqr": float(np.subtract(*np.percentile(r, [75, 25]))),
        f"{prefix}_left_iqr": float(np.subtract(*np.percentile(l, [75, 25]))),
        f"{prefix}_diff_median": float(np.median(r - l)),
    })
    return out


def native_feature_schema() -> dict[str, float]:
    """Fixed feature names with missing defaults (built by running on an empty track)."""
    return {name: (0.0 if name.endswith("_n") else float("nan")) for name in _FEATURE_NAMES}


def native_features(observations: pd.DataFrame, timeline: pd.DataFrame, *, fps: float, test_type: str) -> dict[str, float]:
    result: dict[str, float] = {}
    track = build_track(observations, timeline, fps=fps) if not observations.empty and not timeline.empty else None
    if track is None:
        result.update({name: (0.0 if name.endswith("_n") else float("nan")) for name in _FEATURE_NAMES})
        return result
    scale = track.px_per_pd
    result["nat_q_px_per_pd"] = float(scale)
    result["nat_q_eye_width_px"] = float(track.eye_width_px)
    for eye in ("right", "left"):
        conf = track.pos(eye, "conf")
        result[f"nat_q_{eye}_conf_fraction"] = float(np.mean(conf >= CONF_GATE)) if len(conf) else 0.0
    hm = track.head_motion[np.isfinite(track.head_motion)]
    result["nat_q_head_motion_p90"] = float(np.percentile(hm, 90)) if len(hm) else float("nan")
    yaw = track.yaw[np.isfinite(track.yaw)]
    result["nat_q_yaw_median"] = float(np.median(yaw)) if len(yaw) else float("nan")
    result["nat_q_yaw_iqr"] = float(np.subtract(*np.percentile(yaw, [75, 25]))) if len(yaw) >= 3 else float("nan")

    measures = measure_all_events(track, timeline, test_type=test_type)
    for frame_ in ("corner", "rigid"):
        result.update(_binocular_features(track, frame_, f"nat_bin_{frame_}"))
        for kind in EVENT_KINDS:
            ms = [m for m in measures if m.kind == kind and m.frame_ == frame_]
            long_ms = [m for m in ms if not np.isfinite(m.cover_dur_s) or m.cover_dur_s >= 1.0] if kind == "unc" else ms
            for subset, tag in ((ms, "all"), (long_ms, "long")):
                w = [m.quality for m in subset]
                p = f"nat_{kind}_{frame_}_{tag}"
                result.update(_summary([m.step_amp / scale for m in subset], w, f"{p}_step"))
                result.update(_summary([m.late_vs_first / scale for m in subset], w, f"{p}_latefirst"))
                result.update(_summary([m.deviation_vs_prefix / scale for m in subset], w, f"{p}_devprefix"))
                if tag == "all":
                    result[f"{p}_latency_median_s"] = _median([m.step_latency_s for m in subset])
                    result[f"{p}_stepq_median"] = _median([m.step_quality for m in subset])
                    result[f"{p}_drift_median"] = _median([m.drift_slope / scale for m in subset])
                    result[f"{p}_drift_absmax"] = float(np.max(np.abs(_finite([m.drift_slope / scale for m in subset])))) if len(_finite([m.drift_slope for m in subset])) else float("nan")
                    result[f"{p}_cover_dur_median_s"] = _median([m.cover_dur_s for m in subset])
                    result[f"{p}_cover_dur_max_s"] = float(np.max(_finite([m.cover_dur_s for m in subset]))) if len(_finite([m.cover_dur_s for m in subset])) else float("nan")
                    result[f"{p}_head_motion_median"] = _median([m.head_motion_px for m in subset])
                    result[f"{p}_quality_median"] = _median([m.quality for m in subset])
            # per-eye dominant step (fixation preference / which eye moves)
            for eye in ("right", "left"):
                sub = [m for m in ms if m.observed_eye == eye]
                result[f"nat_{kind}_{frame_}_{eye}_step_median"] = _median([m.step_amp / scale for m in sub])
                result[f"nat_{kind}_{frame_}_{eye}_n"] = float(len(sub))
    # cross-frame agreement (corner vs rigid) as a reliability signal
    for kind in EVENT_KINDS:
        a = result.get(f"nat_{kind}_corner_all_step_median", float("nan"))
        b = result.get(f"nat_{kind}_rigid_all_step_median", float("nan"))
        result[f"nat_{kind}_frame_agreement"] = float(abs(a - b)) if np.isfinite(a) and np.isfinite(b) else float("nan")
    # ensure fixed schema
    for name in _FEATURE_NAMES:
        result.setdefault(name, 0.0 if name.endswith("_n") else float("nan"))
    return result


def _build_feature_names() -> tuple[str, ...]:
    names = ["nat_q_px_per_pd", "nat_q_eye_width_px", "nat_q_right_conf_fraction", "nat_q_left_conf_fraction", "nat_q_head_motion_p90", "nat_q_yaw_median", "nat_q_yaw_iqr"]
    for frame_ in ("corner", "rigid"):
        names += [f"nat_bin_{frame_}_{k}" for k in ("n", "vergence_median", "vergence_iqr", "right_median", "left_median", "right_iqr", "left_iqr", "diff_median")]
        for kind in EVENT_KINDS:
            for tag in ("all", "long"):
                p = f"nat_{kind}_{frame_}_{tag}"
                for stat in ("step", "latefirst", "devprefix"):
                    names += list(_summary([], [], f"{p}_{stat}").keys())
                if tag == "all":
                    names += [f"{p}_{k}" for k in ("latency_median_s", "stepq_median", "drift_median", "drift_absmax", "cover_dur_median_s", "cover_dur_max_s", "head_motion_median", "quality_median")]
            for eye in ("right", "left"):
                names += [f"nat_{kind}_{frame_}_{eye}_step_median", f"nat_{kind}_{frame_}_{eye}_n"]
    for kind in EVENT_KINDS:
        names.append(f"nat_{kind}_frame_agreement")
    return tuple(dict.fromkeys(names))


_FEATURE_NAMES = _build_feature_names()


def native_feature_names() -> tuple[str, ...]:
    return _FEATURE_NAMES


__all__ = ["EventMeasure", "NativeTrack", "build_track", "measure_all_events", "native_feature_names", "native_feature_schema", "native_features"]
