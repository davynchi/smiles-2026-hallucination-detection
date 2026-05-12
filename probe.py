"""
probe.py — Hallucination probe classifier (student-implemented).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from tqdm import tqdm


class HallucinationProbe(nn.Module):
    """Binary classifier: CatBoost on raw features (no PCA).

    Features: max-pool layers 18-22 + direct layers 23,24 + cosine distances
              + geometric features (5427 total).
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._clf = CatBoostClassifier(
            iterations=1000,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=10,
            loss_function="Logloss",
            eval_metric="Accuracy",
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=200,
            random_seed=42,
            verbose=0,
        )
        self._threshold: float = 0.5
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def _build_network(self, input_dim: int) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("forward() is not used; probe uses CatBoost backend.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._X_train = X
        self._y_train = y
        self._clf.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        if self._X_train is not None:
            self._clf.fit(
                self._X_train, self._y_train,
                eval_set=(X_val, y_val),
            )
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_threshold, best_acc = 0.5, -1.0
        for t in tqdm(candidates, desc="Tuning threshold", leave=False, unit="t"):
            acc = float((y_val == (probs >= t).astype(int)).mean())
            if acc > best_acc:
                best_acc, best_threshold = acc, float(t)
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)
