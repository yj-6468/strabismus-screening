"""Unit tests for the native bundle helpers (derived features, decision weights)."""
import numpy as np
import pandas as pd

from strabismus_screen.native_bundle import PHYS_FEATURES, add_derived, bal_acc_weights


def test_add_derived_prefers_long_then_all_then_legacy():
    t = pd.DataFrame({
        "nat_unc_corner_long_latefirst_median": [5.0, np.nan, np.nan],
        "nat_unc_corner_all_latefirst_median": [4.0, 3.0, np.nan],
        "iris_pair_uncover_all_smooth_verylate_median": [0.02, 0.02, 0.02],
        "nat_q_px_per_pd": [0.4, 0.4, 0.4],
        "nat_q_eye_width_px": [100.0, 100.0, 100.0],
        "nat_unc_corner_long_devprefix_median": [np.nan, 1.0, np.nan],
        "nat_unc_corner_all_devprefix_median": [2.0, 2.0, 2.0],
    })
    d = add_derived(t)
    assert d["nat_best_latefirst"].tolist() == [5.0, 3.0, 0.02 * 100.0 / 0.4]
    assert d["nat_best_source"].tolist() == [2.0, 1.0, 0.0]
    assert d["nat_best_devprefix"].tolist() == [2.0, 1.0, 2.0]


def test_bal_acc_weights_improve_minority_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    P = np.c_[0.6 + 0.2 * rng.random(100), 0.4 - 0.2 * rng.random(100)]
    P[y == 1, 1] += 0.15
    P = P / P.sum(1, keepdims=True)
    w = bal_acc_weights(P, y, 2)
    before = np.argmax(P, 1); after = np.argmax(P * w[None, :], 1)
    ba = lambda pred: np.mean([np.mean(pred[y == k] == k) for k in (0, 1)])  # noqa: E731
    assert ba(after) >= ba(before)
    assert w[1] >= 1.0


def test_phys_feature_list_is_unique():
    assert len(PHYS_FEATURES) == len(set(PHYS_FEATURES))
