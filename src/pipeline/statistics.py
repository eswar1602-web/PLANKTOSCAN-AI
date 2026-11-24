"""Count intervals and concentration: what one image's counts mean, with their uncertainty stated.

Two different uncertainties, kept apart:

1. Classification interval (per image, `classification_intervals`): which class each detection belongs
   to is uncertain. With the classifier's probability p_i(c) for every detection, the number of
   class-c organisms is a sum of independent Bernoulli(p_i(c)) variables: a Poisson-binomial
   distribution. Its mean is the "expected" (soft) count and its 2.5% / 97.5% quantiles are the
   interval. It assumes calibrated probabilities and independent errors, and it does NOT cover
   organisms the detector missed or false detections. Its coverage was measured on synthetic VAL
   frames (scripts/check_count_intervals.py -> results/count_intervals_val.json).

2. Sampling interval (for concentration, `concentration`): an image shows a small volume of the sample.
   If organisms are randomly spread, the count in that volume is Poisson; the exact (Garwood) interval
   for the Poisson mean, divided by the volume, is the concentration interval. This is the standard
   counting-chamber statistic (about +-2 sqrt(n) for large n): it does NOT include classification or
   detection errors, and clumped samples have wider true intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2

LEVEL = 0.95
UM3_PER_ML = 1e12            # 1 mL = 1 cm^3 = (10^4 um)^3


# ---------------------------------------------------------------- Poisson (sampling)
def poisson_ci(n: int, level: float = LEVEL) -> tuple[float, float]:
    """Exact (Garwood) two-sided interval for the mean of a Poisson variable observed as n."""
    if n < 0 or int(n) != n:
        raise ValueError(f"count must be a non-negative integer, got {n}")
    a = 1.0 - level
    low = 0.0 if n == 0 else float(chi2.ppf(a / 2, 2 * n) / 2)
    high = float(chi2.ppf(1 - a / 2, 2 * n + 2) / 2)
    return low, high


# ---------------------------------------------------------------- Poisson-binomial (classification)
def poisson_binomial_pmf(ps: Iterable[float]) -> np.ndarray:
    """P(K = k), k = 0..n, for K = sum of independent Bernoulli(p_i). Exact dynamic programming, O(n^2)."""
    pmf = np.array([1.0])
    for p in ps:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of [0, 1]: {p}")
        pmf = np.concatenate([pmf * (1 - p), [0.0]]) + np.concatenate([[0.0], pmf * p])
    return pmf


def soft_count_interval(ps: Iterable[float], level: float = LEVEL) -> dict[str, float | int]:
    """Expected count and equal-tailed interval [low, high] with P(low <= K <= high) >= level."""
    ps = list(ps)
    cdf = np.cumsum(poisson_binomial_pmf(ps))
    a = (1.0 - level) / 2
    eps = 1e-12                                       # float round-off in the cumulative sum
    low = int(np.searchsorted(cdf, a - eps, side="right"))      # first k with CDF(k) > a
    high = int(np.searchsorted(cdf, 1 - a - eps, side="left"))  # first k with CDF(k) >= 1 - a
    return {"expected": round(float(sum(ps)), 3), "low": low, "high": min(high, len(ps))}


def classification_intervals(dets: list, classes: list[str], level: float = LEVEL,
                             calibrated: bool = False) -> dict[str, Any]:
    """Per-class expected count and interval from the classifier probabilities of every detection.

    `classes` is the full classifier output list (organism classes and the reject class); every
    detection contributes its probability to every class, so the expected counts sum to the number
    of detections. calibrated=True uses the temperature-scaled probabilities (Detection.calibrated_probs).
    """
    def p(d, c):
        return (d.calibrated_probs if calibrated else d.probs).get(c, 0.0)

    per_class = {c: soft_count_interval([p(d, c) for d in dets], level) for c in classes}
    return {"level": level, "per_class": per_class, "probabilities": "calibrated" if calibrated else "raw",
            "method": "Poisson-binomial over the classifier probabilities of all detections; "
                      "assumes calibrated probabilities, does not include missed or false detections"}


# ---------------------------------------------------------------- concentration
@dataclass(frozen=True)
class Calibration:
    """How much sample one image shows. Give imaged_volume_ml, or pixel_size_um together with depth_um.

    concentration_factor: how many times the sample was concentrated before imaging (e.g. 1 L reduced to
    10 mL = 100). Concentrations are reported for the ORIGINAL water: count / (imaged volume x factor).
    """
    imaged_volume_ml: float | None = None
    pixel_size_um: float | None = None
    depth_um: float | None = None
    concentration_factor: float = 1.0

    def __post_init__(self):
        for name in ("imaged_volume_ml", "pixel_size_um", "depth_um", "concentration_factor"):
            v = getattr(self, name)
            if v is not None and not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive number, got {v}")
        has_volume = self.imaged_volume_ml is not None
        has_geometry = self.pixel_size_um is not None or self.depth_um is not None
        if has_volume and has_geometry:
            raise ValueError("give either imaged_volume_ml or pixel_size_um + depth_um, not both")
        if not has_volume and (self.pixel_size_um is None or self.depth_um is None):
            raise ValueError("calibration needs imaged_volume_ml, or both pixel_size_um and depth_um")

    def volume_ml(self, width_px: int, height_px: int) -> float:
        """Volume of sample shown in a width x height image, in mL (before the concentration factor)."""
        if self.imaged_volume_ml is not None:
            return float(self.imaged_volume_ml)
        return width_px * height_px * self.pixel_size_um ** 2 * self.depth_um / UM3_PER_ML

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def concentration(counts: dict[str, int], volume_ml: float, factor: float = 1.0,
                  level: float = LEVEL) -> dict[str, dict[str, float | int]]:
    """Per class: count, organisms per mL and per litre of the original water, and the Poisson interval."""
    if not (volume_ml > 0 and factor > 0):
        raise ValueError("volume and concentration factor must be positive")
    water_ml = volume_ml * factor
    out = {}
    for c, n in counts.items():
        lo, hi = poisson_ci(int(n), level)
        out[c] = {"count": int(n), "per_ml": n / water_ml, "per_ml_low": lo / water_ml, "per_ml_high": hi / water_ml,
                  "per_l": 1000 * n / water_ml, "per_l_low": 1000 * lo / water_ml, "per_l_high": 1000 * hi / water_ml}
    return out


def concentration_block(confirmed: dict[str, int], calib: Calibration, width_px: int, height_px: int,
                        level: float = LEVEL) -> dict[str, Any]:
    """The canonical 'concentration' entry of a result: inputs, volumes and per-class concentrations."""
    vol = calib.volume_ml(width_px, height_px)
    total = {"all organisms": sum(confirmed.values())}
    return {"calibration": calib.to_dict(), "imaged_volume_ml": vol,
            "water_volume_ml": vol * calib.concentration_factor, "level": level,
            "per_class": concentration(confirmed, vol, calib.concentration_factor, level),
            "total": concentration(total, vol, calib.concentration_factor, level)["all organisms"],
            "counts_used": "confirmed",
            "method": "exact Poisson (Garwood) interval on the confirmed count, divided by the water volume; "
                      "sampling error only, assumes randomly spread organisms"}


def pooled_concentration(results: list[dict[str, Any]], level: float = LEVEL) -> dict[str, Any] | None:
    """Concentration over several analysed images of one sample: summed counts over summed water volume.

    Uses only results that have a concentration block; None if there are none. Pooling many images
    gives a much narrower interval than any single image.
    """
    blocks = [r["concentration"] for r in results if r.get("concentration")]
    if not blocks:
        return None
    classes = list(blocks[0]["per_class"])
    counts = {c: sum(b["per_class"][c]["count"] for b in blocks) for c in classes}
    water = sum(b["water_volume_ml"] for b in blocks)
    per_class = concentration(counts, water, 1.0, level)
    return {"images": len(blocks), "water_volume_ml": water, "level": level, "per_class": per_class,
            "total": concentration({"t": sum(counts.values())}, water, 1.0, level)["t"], "counts_used": "confirmed"}
