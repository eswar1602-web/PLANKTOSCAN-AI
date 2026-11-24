"""Calibrated confidence and the 'possibly unknown' flag, fitted once on known data (no weights change).

Temperature scaling (Guo et al. 2017): one number T, fitted on VAL by minimising the negative
log-likelihood, rescales the logits: p = softmax(z / T). The argmax never changes. The classifier was
trained with label smoothing 0.1 over 8 classes, which caps a confident softmax near 0.91, so the raw
probabilities are not probabilities of being right; calibrated ones are closer to that.

Unknown flag (Lee et al. 2018): the classifier's penultimate features (1280-d) of the TRAIN crops give a
mean per class and one shared covariance (Ledoit-Wolf shrinkage). A crop's score is its smallest squared
Mahalanobis distance to a class mean. The cut-off is a quantile of the scores of KNOWN VAL crops (e.g.
95% -> about 5% of known organisms flagged), so it is never tuned on unknown organisms. Flagged
detections keep their status and counts; the flag only says "this looks unlike everything trained on".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


# ---------------------------------------------------------------- temperature
def nll(logits: np.ndarray, y: np.ndarray, t: float = 1.0) -> float:
    """Mean negative log-likelihood of the true labels at temperature t."""
    return float(-log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the NLL (bounded search, log scale)."""
    res = minimize_scalar(lambda lt: nll(logits, y, float(np.exp(lt))), bounds=(np.log(0.05), np.log(20)),
                          method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


# ---------------------------------------------------------------- Mahalanobis
@dataclass
class Mahalanobis:
    means: np.ndarray        # (C, D) class means of the train features
    precision: np.ndarray    # (D, D) inverse of the shared (shrunk) covariance

    @classmethod
    def fit(cls, feats: np.ndarray, y: np.ndarray, n_classes: int) -> "Mahalanobis":
        from sklearn.covariance import LedoitWolf
        feats = feats.astype(np.float64)
        means = np.stack([feats[y == c].mean(0) for c in range(n_classes)])
        centred = feats - means[y]
        return cls(means=means, precision=LedoitWolf(assume_centered=True).fit(centred).precision_)

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Smallest squared Mahalanobis distance to any class mean, per row."""
        f = feats.astype(np.float64)
        # d^2 = f P f - 2 f P m + m P m, for every class at once
        fp = f @ self.precision
        d2 = (fp * f).sum(1)[:, None] - 2 * fp @ self.means.T + ((self.means @ self.precision) * self.means).sum(1)[None]
        return np.maximum(d2.min(1), 0.0)


# ---------------------------------------------------------------- the deployed model
@dataclass
class ConfidenceModel:
    temperature: float
    ood: Mahalanobis
    ood_threshold: float
    classifier_run: str
    info: dict[str, Any]

    def apply(self, logits: np.ndarray, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(calibrated probabilities (N, C), OOD scores (N,), unknown flags (N,))."""
        if len(logits) == 0:
            return np.zeros((0, logits.shape[1] if logits.ndim == 2 else 0)), np.zeros(0), np.zeros(0, bool)
        scores = self.ood.score(feats)
        return softmax(logits / self.temperature, axis=1), scores, scores > self.ood_threshold

    def save(self, path: Path) -> None:
        import json
        np.savez_compressed(path, temperature=self.temperature, means=self.ood.means, precision=self.ood.precision,
                            ood_threshold=self.ood_threshold, classifier_run=self.classifier_run,
                            info=json.dumps(self.info))

    @classmethod
    def load(cls, path: Path) -> "ConfidenceModel":
        import json
        z = np.load(path, allow_pickle=False)
        return cls(temperature=float(z["temperature"]), ood=Mahalanobis(z["means"], z["precision"]),
                   ood_threshold=float(z["ood_threshold"]), classifier_run=str(z["classifier_run"]),
                   info=json.loads(str(z["info"])))
