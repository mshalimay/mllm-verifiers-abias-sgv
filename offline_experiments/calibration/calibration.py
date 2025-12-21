"""
calibration.py

Calibration utilities:
    - Platt scaling    (via sklearn.CalibratedClassifierCV, method="sigmoid")
    - Isotonic regression
    - Histogram calibration
    - Logistic regression calibration
    - Prior-shift correction

Usage:
    from calibration import Calibrator

    cal = Calibrator(method="platt")         # or isotonic, histogram, logistic, prior
    cal.fit(scores_train, labels_train)
    p = cal.predict(scores_test)
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# sklearn imports
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    # New in scikit-learn >=1.6: FrozenEstimator replaces cv='prefit'
    try:
        from sklearn.base import FrozenEstimator  # type: ignore

        _HAS_FROZEN = True
    except Exception:
        _HAS_FROZEN = False
    _HAS_SKLEARN = True
except Exception:
    # If sklearn (or submodules) aren't available, fall back gracefully.
    _HAS_SKLEARN = False
    _HAS_FROZEN = False

# (Optional) If sklearn missing, runtime paths that require it will raise.


# ============================================================
# For Platt parameters container (only for reporting, optional)
# ============================================================


@dataclass
class PlattParams:
    A: float
    B: float


# ============================================================
# Calibrator
# ============================================================


class Calibrator:
    """
    Unified calibration wrapper for:

      - Platt scaling            (method="platt")
      - Isotonic regression      (method="isotonic")
      - Histogram calibration    (method="histogram")
      - Logistic regression      (method="logistic")
      - Prior-shift correction   (method="prior")

    API:
        cal = Calibrator(method="platt")
        cal.fit(scores, labels)
        p = cal.predict(scores)
    """

    def __init__(
        self,
        method: str = "platt",
        max_iter: int = 200,
    ):
        assert method in {
            "platt",
            "isotonic",
            "histogram",
            "logistic",
            "prior",
        }, f"Invalid method {method}"

        self.method = method
        self.max_iter = max_iter

        # For sklearn-Platt
        self.platt_model: Optional[Any] = None
        self.platt_params: Optional[PlattParams] = None  # For reporting only

        # Isotonic model
        self.iso_model: Optional[Any] = None

        # Histogram
        self.hist_scores: Optional[np.ndarray] = None
        self.hist_probs: Optional[np.ndarray] = None

        # Logistic regression calibration
        self.lr_model: Optional[Any] = None

        # Prior-shift
        self.prior_target: Optional[float] = None
        self.prior_logit_shift: Optional[float] = None

    # ============================================================
    # Sigmoid helper
    # ============================================================

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    # ============================================================
    # Platt scaling (SKLEARN IMPLEMENTATION)
    # ============================================================

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError("Platt scaling requires scikit-learn.")

        # Use a dummy logistic-regression model; only feature is score
        base = LogisticRegression(  # type: ignore
            solver="lbfgs",
            max_iter=self.max_iter,
            class_weight="balanced",
        )

        X = scores.reshape(-1, 1)

        # Fit LR first
        base.fit(X, labels)

        # Then calibrate its decision function using sigmoid = Platt.
        # Forward-compatible path (sklearn >=1.6) avoids deprecated cv='prefit'.
        if _HAS_FROZEN:
            calibrated = CalibratedClassifierCV(  # type: ignore
                estimator=FrozenEstimator(base),  # base already fitted  # type: ignore
                method="sigmoid",
            )
        else:
            # Backward-compatible legacy path for sklearn <1.6
            calibrated = CalibratedClassifierCV(  # type: ignore
                estimator=base,
                method="sigmoid",
                cv="prefit",
            )
        calibrated.fit(X, labels)

        self.platt_model = calibrated

        # Optional: extract A and B (for reporting)
        # sklearn stores calibrated.classifiers_[0].classes_, calibrated.calibrators_
        # But A,B are not exposed directly; we compute a least squares logit fit:
        eps = 1e-6
        p = calibrated.predict_proba(X)[:, 1]
        logit_p = np.log(p.clip(eps, 1 - eps) / (1 - p).clip(eps, 1 - eps))
        A, B = np.polyfit(scores, logit_p, deg=1)
        self.platt_params = PlattParams(float(A), float(B))

    def _predict_platt(self, scores: np.ndarray) -> np.ndarray:
        if self.platt_model is None:
            raise RuntimeError("Platt calibrator is not fitted.")
        X = scores.reshape(-1, 1)
        return self.platt_model.predict_proba(X)[:, 1]

    # ============================================================
    # Isotonic regression
    # ============================================================

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError("Isotonic regression requires sklearn.")
        self.iso_model = IsotonicRegression(out_of_bounds="clip")  # type: ignore
        self.iso_model.fit(scores, labels)

    def _predict_isotonic(self, scores: np.ndarray) -> np.ndarray:
        if self.iso_model is None:
            raise RuntimeError("Isotonic calibrator is not fitted.")
        return self.iso_model.predict(scores)

    # ============================================================
    # Histogram calibration
    # ============================================================

    def _fit_histogram(self, scores: np.ndarray, labels: np.ndarray) -> None:
        uniq = np.unique(scores)
        self.hist_scores = uniq
        self.hist_probs = np.zeros_like(uniq, float)
        for i, v in enumerate(uniq):
            mask = scores == v
            self.hist_probs[i] = labels[mask].mean()

    def _predict_histogram(self, scores: np.ndarray) -> np.ndarray:
        if self.hist_scores is None or self.hist_probs is None:
            raise RuntimeError("Histogram calibrator is not fitted.")
        idx = np.searchsorted(self.hist_scores, scores, side="right") - 1
        idx = np.clip(idx, 0, len(self.hist_probs) - 1)
        return self.hist_probs[idx]

    # ============================================================
    # Logistic regression calibration
    # ============================================================

    def _fit_logistic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError("Logistic calibration requires sklearn.")
        X = scores.reshape(-1, 1)
        model = LogisticRegression(  # type: ignore
            solver="lbfgs",
            class_weight="balanced",
            max_iter=self.max_iter,
        )
        model.fit(X, labels)
        self.lr_model = model

    def _predict_logistic(self, scores: np.ndarray) -> np.ndarray:
        if self.lr_model is None:
            raise RuntimeError("Logistic calibrator is not fitted.")
        X = scores.reshape(-1, 1)
        return self.lr_model.predict_proba(X)[:, 1]

    # ============================================================
    # Prior-shift correction
    # ============================================================

    def _fit_prior(self, scores: np.ndarray, labels: np.ndarray, pi_star=None) -> None:
        pi = labels.mean()
        if pi_star is None:
            pi_star = 0.5  # target prior
        self.prior_target = pi_star

        eps = 1e-9
        shift = np.log(pi_star / (1 - pi_star + eps)) - np.log(pi / (1 - pi + eps))
        self.prior_logit_shift = float(shift)

    def _predict_prior(self, scores: np.ndarray) -> np.ndarray:
        p = np.clip(scores.astype(float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        logit2 = logit + self.prior_logit_shift
        return 1 / (1 + np.exp(-logit2))

    # ============================================================
    # Public API
    # ============================================================

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "Calibrator":
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, float)

        if self.method == "platt":
            self._fit_platt(scores, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(scores, labels)
        elif self.method == "histogram":
            self._fit_histogram(scores, labels)
        elif self.method == "logistic":
            self._fit_logistic(scores, labels)
        elif self.method == "prior":
            self._fit_prior(scores, labels)
        else:
            raise ValueError(f"Unknown method {self.method}")

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, float)

        if self.method == "platt":
            return self._predict_platt(scores)
        if self.method == "isotonic":
            return self._predict_isotonic(scores)
        if self.method == "histogram":
            return self._predict_histogram(scores)
        if self.method == "logistic":
            return self._predict_logistic(scores)
        if self.method == "prior":
            return self._predict_prior(scores)

        raise ValueError(f"Unknown method {self.method}")
