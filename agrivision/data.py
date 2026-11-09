"""Deterministic, domain-specific synthetic dataset for AgriVision-Crop — Synthetic Crop Suitability Classification."""

from __future__ import annotations
import numpy as np

FEATURES = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
CLASS_NAMES = ["rice", "wheat", "maize", "cotton", "banana", "coffee"]
GROUP_NAME = None
GROUP_SPLIT = False

def make_dataset(n=2400, seed=42):
    if n < 80:
        raise ValueError("n must be at least 80 for a useful holdout")
    rng = np.random.default_rng(seed)
    class_ids = rng.integers(0, len(FEATURES), n)  # replaced below with crop-specific profiles
    profiles = np.array([[90,45,40,24,82,6.5,230],[55,50,30,18,55,6.8,95],[80,42,25,23,62,6.3,110],[112,42,25,28,70,6.5,85],[100,80,52,27,80,6.2,120],[100,32,30,24,65,6.0,170]], dtype=float)
    scales = np.array([[18,10,9,2.5,6,.4,35],[15,10,8,2.5,7,.4,25],[16,9,7,2.5,7,.4,28],[18,9,7,2.5,7,.4,24],[16,12,10,2,6,.4,28],[16,9,8,2.5,7,.4,32]], dtype=float)
    y = rng.integers(0, len(profiles), n)
    X = rng.normal(profiles[y], scales[y])
    X[:, [0,1,2,6]] = np.clip(X[:, [0,1,2,6]], 0, None)
    X[:, 5] = np.clip(X[:, 5], 3.5, 9.5)
    groups = np.full(n, "synthetic", dtype=object)
    nitrogen, phosphorus, potassium = X[:, 0], X[:, 1], X[:, 2]
    temperature, humidity, ph, rainfall = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
    X = np.column_stack([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).astype(float)
    if X.shape != (n, len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("generator produced invalid feature matrix")
    if np.asarray(groups).shape != (n,):
        raise ValueError("generator produced invalid group vector")
    return X, y.astype(int), np.asarray(groups)
