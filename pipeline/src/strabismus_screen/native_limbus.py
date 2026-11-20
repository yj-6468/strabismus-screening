"""Native-resolution, sub-pixel limbus localization for one eye crop.

The Face Landmarker's iris points are computed on an internally downsampled
64x64 eye crop, so they cannot resolve the ~0.45 px per prism diopter of
limbus translation at this archive's ~46 px iris.  This module measures the
limbus directly on the native frame pixels:

* CLAHE-enhanced luma gradient field (no resampling of the image);
* Daugman-style integro-differential score restricted to the two lateral
  sectors (+-35 degrees about horizontal) so eyelids do not bias the fit;
* exhaustive coarse search over centre and radius, 0.25 px refinement, then
  parabolic sub-pixel interpolation of the smooth score surface;
* one-sided fallback with the radius pinned to the clip prior when one limbus
  edge is hidden by the occluder or the eye is only partially visible;
* an absolute raw-luma contrast test (sclera brighter than iris) so smooth
  occluders and reflections cannot produce a confident false circle.

A small per-eye tracker seeds the search from the previous accepted fit when
the landmark seed disagrees with it, and keeps a running radius prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


SECTOR_DEG = 35.0
SECTOR_STEP_DEG = 4.0
STATUS_CODES = {"failed": 0, "both": 1, "right_only": 2, "left_only": 3}
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _prep(crop_bgr: np.ndarray):
    raw = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = _CLAHE.apply(raw)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1) / 32.0
    scale = float(np.percentile(np.hypot(gx, gy), 90)) + 1e-6
    return gx, gy, scale, cv2.GaussianBlur(raw, (0, 0), 1.0).astype(np.float32)


def _sample(field_: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = field_.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    v = (
        field_[y0c, x0c] * (1 - fx) * (1 - fy)
        + field_[y0c, x0c + 1] * fx * (1 - fy)
        + field_[y0c + 1, x0c] * (1 - fx) * fy
        + field_[y0c + 1, x0c + 1] * fx * fy
    )
    return np.where(valid, v, 0.0)


def _side_angles(side: str) -> np.ndarray:
    base = 0.0 if side == "right" else np.pi
    offs = np.deg2rad(np.arange(-SECTOR_DEG, SECTOR_DEG + 1e-6, SECTOR_STEP_DEG))
    return base + offs


_ANG_R = _side_angles("right")
_ANG_L = _side_angles("left")
_ANG_BOTH = np.concatenate([_ANG_R, _ANG_L])


def _scores(gx, gy, scale, cxs, cys, rs, angles, mask_fn):
    cxs = np.asarray(cxs, float).ravel()
    cys = np.asarray(cys, float).ravel()
    rs = np.asarray(rs, float).ravel()
    ca = np.cos(angles)[None, :]
    sa = np.sin(angles)[None, :]
    xs = cxs[:, None] + rs[:, None] * ca
    ys = cys[:, None] + rs[:, None] * sa
    radial = _sample(gx, xs, ys) * ca + _sample(gy, xs, ys) * sa
    weight = mask_fn(xs, ys)
    wsum = np.maximum(weight.sum(axis=1), 1e-6)
    return (radial * weight).sum(axis=1) / wsum / scale, weight.mean(axis=1)


def _contour_mask_fn(contour: np.ndarray | None, shape):
    if contour is None or len(contour) < 3:
        return lambda xs, ys: np.ones_like(xs)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(contour).astype(np.int32)], 1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    return lambda xs, ys: (_sample(mask, xs, ys) > 0.5).astype(np.float32)


def _ring_contrast(raw, cx, cy, r, angles) -> float:
    ca, sa = np.cos(angles), np.sin(angles)
    inside = _sample(raw, cx + (r - 3.0) * ca, cy + (r - 3.0) * sa)
    outside = _sample(raw, cx + (r + 3.0) * ca, cy + (r + 3.0) * sa)
    return float(np.median(outside - inside))


def localize_limbus(
    crop_bgr: np.ndarray,
    *,
    cx0: float,
    cy0: float,
    r0: float,
    r_prior: float | None = None,
    contour: np.ndarray | None = None,
    search_x: int = 10,
    search_y: int = 6,
) -> dict[str, float | str]:
    """Fit a limbus circle near the seed.  All coordinates are crop pixels."""

    out: dict[str, float | str] = dict(
        cx=math.nan, cy=math.nan, r=math.nan, conf=0.0, step_luma=math.nan,
        temporal_score=math.nan, nasal_score=math.nan, status="failed",
    )
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 16:
        return out
    if not all(np.isfinite(v) for v in (cx0, cy0, r0)) or r0 < 6:
        return out
    if r_prior is None or not np.isfinite(r_prior):
        r_prior = r0
    gx, gy, scale, raw = _prep(crop_bgr)
    mask_fn = _contour_mask_fn(contour, gx.shape)

    dcx = np.arange(-search_x, search_x + 1, 1.0)
    dcy = np.arange(-search_y, search_y + 1, 2.0)
    rr = np.arange(np.floor(0.80 * r_prior), np.ceil(1.25 * r_prior) + 0.5, 1.5)
    CX, CY, R = np.meshgrid(cx0 + dcx, cy0 + dcy, rr, indexing="ij")
    s_all, cov = _scores(gx, gy, scale, CX, CY, R, _ANG_BOTH, mask_fn)
    s_r, _ = _scores(gx, gy, scale, CX, CY, R, _ANG_R, mask_fn)
    s_l, _ = _scores(gx, gy, scale, CX, CY, R, _ANG_L, mask_fn)
    score = s_all * np.minimum(1.0, cov / 0.5)
    k = int(np.argmax(score))
    bcx, bcy, br = float(CX.ravel()[k]), float(CY.ravel()[k]), float(R.ravel()[k])
    side_r, side_l = float(s_r[k]), float(s_l[k])

    status = "both"
    weak = min(side_r, side_l) < 0.35 * max(side_r, side_l) or min(side_r, side_l) < 0.05
    if weak and max(side_r, side_l) > 0.08:
        good = _ANG_R if side_r >= side_l else _ANG_L
        status = "right_only" if side_r >= side_l else "left_only"
        dcx2 = np.arange(-search_x - 2, search_x + 3, 1.0)
        dcy2 = np.arange(-4, 5, 1.0)
        CX2, CY2 = np.meshgrid(cx0 + dcx2, cy0 + dcy2, indexing="ij")
        R2 = np.full_like(CX2, r_prior)
        s2, cov2 = _scores(gx, gy, scale, CX2, CY2, R2, good, mask_fn)
        k2 = int(np.argmax(s2 * np.minimum(1.0, cov2 / 0.5)))
        bcx, bcy, br = float(CX2.ravel()[k2]), float(CY2.ravel()[k2]), float(r_prior)
        ang_ref = good
    else:
        ang_ref = _ANG_BOTH

    fx = np.arange(-1.0, 1.01, 0.25)
    fy = np.arange(-1.5, 1.51, 0.5)
    fr_ = np.arange(-1.5, 1.51, 0.5) if status == "both" else np.array([0.0])
    FX, FY, FR = np.meshgrid(bcx + fx, bcy + fy, br + fr_, indexing="ij")
    sf, covf = _scores(gx, gy, scale, FX, FY, FR, ang_ref, mask_fn)
    sf = sf * np.minimum(1.0, covf / 0.5)
    kf = int(np.argmax(sf))
    bcx, bcy, br = float(FX.ravel()[kf]), float(FY.ravel()[kf]), float(FR.ravel()[kf])
    best = float(sf[kf])

    xs = bcx + np.array([-0.25, 0.0, 0.25])
    s3, _ = _scores(gx, gy, scale, xs, np.full(3, bcy), np.full(3, br), ang_ref, mask_fn)
    denom = s3[0] - 2 * s3[1] + s3[2]
    if denom < -1e-9:
        bcx += 0.25 * 0.5 * (s3[0] - s3[2]) / denom
    ys = bcy + np.array([-0.25, 0.0, 0.25])
    s3y, _ = _scores(gx, gy, scale, np.full(3, bcx), ys, np.full(3, br), ang_ref, mask_fn)
    denom = s3y[0] - 2 * s3y[1] + s3y[2]
    if denom < -1e-9:
        bcy += 0.25 * 0.5 * (s3y[0] - s3y[2]) / denom

    sr, _ = _scores(gx, gy, scale, [bcx], [bcy], [br], _ANG_R, mask_fn)
    sl, _ = _scores(gx, gy, scale, [bcx], [bcy], [br], _ANG_L, mask_fn)
    step_r = _ring_contrast(raw, bcx, bcy, br, _ANG_R)
    step_l = _ring_contrast(raw, bcx, bcy, br, _ANG_L)
    step = max(step_r, step_l) if status != "both" else min(step_r, step_l)
    contrast_conf = float(np.clip((step - 6.0) / 24.0, 0.0, 1.0))
    out.update(
        cx=bcx, cy=bcy, r=br,
        conf=float(np.clip(best / 0.6, 0.0, 1.0)) * contrast_conf,
        step_luma=float(step),
        temporal_score=float(sr[0]), nasal_score=float(sl[0]),
        status=status,
    )
    return out


@dataclass
class LimbusTracker:
    """Per-eye temporal state: radius prior and last accepted centre (frame coords)."""

    accept_conf: float = 0.30
    max_gap_frames: int = 3
    radii: list[float] = field(default_factory=list)
    last_frame: int = -10_000
    last_cx: float = math.nan
    last_cy: float = math.nan

    @property
    def r_prior(self) -> float:
        if len(self.radii) < 3:
            return math.nan
        return float(np.median(self.radii[-60:]))

    def seed(self, frame: int, mp_cx: float, mp_cy: float) -> tuple[float, float]:
        """Prefer the previous accepted native centre when it is recent and near the landmark seed."""
        if (
            frame - self.last_frame <= self.max_gap_frames
            and np.isfinite(self.last_cx)
            and abs(self.last_cx - mp_cx) <= 12.0
            and abs(self.last_cy - mp_cy) <= 8.0
        ):
            return 0.5 * (self.last_cx + mp_cx), 0.5 * (self.last_cy + mp_cy)
        return mp_cx, mp_cy

    def update(self, frame: int, result: dict[str, float | str], ox: float, oy: float) -> None:
        if float(result["conf"]) >= self.accept_conf and result["status"] == "both":
            self.radii.append(float(result["r"]))
        if float(result["conf"]) >= self.accept_conf:
            self.last_frame = int(frame)
            self.last_cx = float(result["cx"]) + ox
            self.last_cy = float(result["cy"]) + oy


__all__ = ["LimbusTracker", "STATUS_CODES", "localize_limbus"]
