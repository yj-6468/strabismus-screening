"""Horizontal strabismus video screening research package."""

import os

# The estimators are explicitly single-process.  Some managed macOS sessions
# expose an empty physical-core probe result to joblib, so establish a stable
# logical-core fallback before any submodule imports scikit-learn.
if not os.environ.get("LOKY_MAX_CPU_COUNT", "").strip():
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

from .types import (
    ClinicalClass,
    DirectionClass,
    Eye,
    ObservableState,
    TestType,
)

__all__ = [
    "ClinicalClass",
    "DirectionClass",
    "Eye",
    "ObservableState",
    "TestType",
]

__version__ = "0.1.0"
