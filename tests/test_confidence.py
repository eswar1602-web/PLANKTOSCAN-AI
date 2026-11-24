"""Temperature scaling, the Mahalanobis unknown flag, threshold sensitivity, and their place in results."""

import copy

import numpy as np
import pytest
from scipy.special import softmax

from src.common.config import load_config
from src.pipeline.confidence import ConfidenceModel, Mahalanobis, ece, fit_temperature, nll
from src.service import AnalysisSettings, threshold_sensitivity
from tests.fakes import FakePipeline, fake_service, write_image


# ---------------- temperature
def test_fit_temperature_recovers_the_true_temperature() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, (20000, 8))
    y = np.array([rng.choice(8, p=p) for p in softmax(z / 2.0, 1)])      # labels drawn at temperature 2
    t = fit_temperature(z, y)
    assert t == pytest.approx(2.0, rel=0.05)
    assert nll(z, y, t) < nll(z, y, 1.0)


def test_temperature_never_changes_the_predicted_class() -> None:
    z = np.random.default_rng(1).normal(0, 2, (500, 8))
    for t in (0.3, 0.69, 1.0, 4.0):
        assert (softmax(z / t, 1).argmax(1) == z.argmax(1)).all()


def test_ece_is_small_when_calibrated_and_large_when_overconfident() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, (20000, 4))
    p = softmax(z, 1)
    y = np.array([rng.choice(4, p=q) for q in p])
    assert ece(p, y) < 0.02
    assert ece(softmax(z * 4, 1), y) > 0.1


# ---------------- Mahalanobis
def _clusters(rng, n=400, d=6):
    means = np.eye(3, d) * 10
    x = np.concatenate([rng.normal(m, 1.0, (n, d)) for m in means])
    return x, np.repeat(np.arange(3), n)


def test_mahalanobis_matches_the_brute_force_formula() -> None:
    rng = np.random.default_rng(3)
    x, y = _clusters(rng)
    m = Mahalanobis.fit(x, y, 3)
    q = rng.normal(0, 4, (10, 6))
    brute = np.array([min((v - mu) @ m.precision @ (v - mu) for mu in m.means) for v in q])
    assert np.allclose(m.score(q), brute)


def test_far_points_score_higher_than_known_points() -> None:
    rng = np.random.default_rng(4)
    x, y = _clusters(rng)
    m = Mahalanobis.fit(x, y, 3)
    known = m.score(rng.normal(np.eye(3, 6)[1] * 10, 1.0, (200, 6)))
    far = m.score(rng.normal(-15, 1.0, (200, 6)))
    cut = np.quantile(m.score(x), 0.95)
    assert (known > cut).mean() < 0.15 and (far > cut).mean() == 1.0


def test_confidence_model_round_trip_and_apply(tmp_path) -> None:
    rng = np.random.default_rng(5)
    x, y = _clusters(rng)
    cm = ConfidenceModel(0.7, Mahalanobis.fit(x, y, 3), 30.0, "run-x", {"fitted": "t"})
    cm.save(tmp_path / "c.npz")
    back = ConfidenceModel.load(tmp_path / "c.npz")
    assert back.temperature == 0.7 and back.classifier_run == "run-x" and back.info == {"fitted": "t"}
    logits = rng.normal(0, 2, (4, 3))
    feats = np.concatenate([x[:2], np.full((2, 6), -20.0)])
    p, s, unk = back.apply(logits, feats)
    assert np.allclose(p, softmax(logits / 0.7, 1)) and unk.tolist() == [False, False, True, True]
    p0, s0, u0 = back.apply(np.zeros((0, 3)), np.zeros((0, 6)))
    assert p0.shape == (0, 3) and len(s0) == len(u0) == 0


def test_pipeline_ignores_statistics_fitted_for_another_classifier(tmp_path) -> None:
    from src.pipeline.pipeline import Pipeline
    rng = np.random.default_rng(6)
    x, y = _clusters(rng)
    ConfidenceModel(0.7, Mahalanobis.fit(x, y, 3), 30.0, "old-run", {}).save(tmp_path / "confidence.npz")
    p = Pipeline.__new__(Pipeline)
    p.meta = {"run": "new-run"}
    cm, warn = p._load_confidence(tmp_path / "confidence.npz")
    assert cm is None and "old-run" in warn
    p.meta = {"run": "old-run"}
    cm, warn = p._load_confidence(tmp_path / "confidence.npz")
    assert cm is not None and warn is None
    assert p._load_confidence(tmp_path / "missing.npz")[0] is None


# ---------------- threshold sensitivity
def test_threshold_sensitivity_counts_at_three_thresholds(tmp_path) -> None:
    svc = fake_service(tmp_path, preds=[("copepod", 0.885), ("diatom", 0.895), ("rotifer", 0.95), ("reject", 0.99)])
    dets = svc.pipe.run(None, uncertainty=0.89)
    s = threshold_sensitivity(dets, svc.pipe.organisms, "reject", 0.89, 0.01)
    assert (s["lower"]["confirmed_total"], s["chosen"]["confirmed_total"], s["higher"]["confirmed_total"]) == (3, 2, 1)
    assert s["lower"]["confirmed"]["copepod"] == 1 and s["chosen"]["confirmed"]["copepod"] == 0
    assert s["borderline"] == 2


def test_borderline_warning_and_result_block(tmp_path) -> None:
    svc = fake_service(tmp_path, preds=[("copepod", 0.885), ("diatom", 0.895), ("rotifer", 0.95)])
    r = svc.analyze(write_image(tmp_path / "a.png")).to_dict()
    s = r["threshold_sensitivity"]
    assert s["chosen"]["confirmed_total"] == r["counts"]["confirmed_total"]
    assert any("within 0.01 of the uncertainty threshold" in w for w in r["warnings"])
    (tmp_path / "b").mkdir()
    calm = fake_service(tmp_path / "b", preds=[("copepod", 0.99)]).analyze(write_image(tmp_path / "b.png")).to_dict()
    assert not any("uncertainty threshold:" in w for w in calm["warnings"])


def test_sensitivity_follows_a_custom_threshold(tmp_path) -> None:
    svc = fake_service(tmp_path, preds=[("copepod", 0.65), ("diatom", 0.75)])
    r = svc.analyze(write_image(tmp_path / "a.png"), AnalysisSettings(uncertainty=0.7)).to_dict()
    s = r["threshold_sensitivity"]
    assert s["chosen"]["threshold"] == 0.7 and s["chosen"]["confirmed_total"] == 1


# ---------------- calibrated pipeline output in the service and reports
class CalibratedFake(FakePipeline):
    """FakePipeline plus what the real pipeline adds when models/confidence.npz is loaded."""
    confidence = None

    def run(self, *a, **k):
        dets = super().run(*a, **k)
        for i, d in enumerate(dets):
            d.calibrated_probs = {c: (0.98 if c == d.label else 0.02 / 7) for c in d.probs}
            d.calibrated_prob = 0.98
            d.unknown, d.ood_score = i == 0, 9000.0 + 5000 * (i == 0)
        return dets


def _calibrated_service(tmp_path):
    from src.service import AnalysisService
    det, cls = tmp_path / "det.pt", tmp_path / "cls.pt"
    det.write_bytes(b"d"), cls.write_bytes(b"c")
    cfg = copy.deepcopy(load_config())
    return AnalysisService(cfg, pipeline=CalibratedFake(), detector_path=det, classifier_path=cls, warmup=False)


def test_unknown_flags_are_counted_and_warned_but_do_not_change_counts(tmp_path) -> None:
    (tmp_path / "p").mkdir()
    plain = fake_service(tmp_path / "p").analyze(write_image(tmp_path / "a.png")).to_dict()
    r = _calibrated_service(tmp_path).analyze(write_image(tmp_path / "a.png")).to_dict()
    assert r["counts"]["confirmed"] == plain["counts"]["confirmed"]          # the flag never moves a count
    assert r["counts"]["possibly_unknown"] == 1 and r["counts"]["possibly_unknown_confirmed"] == 1
    assert any("possibly unknown organisms" in w for w in r["warnings"])
    assert r["detections"][0]["unknown"] is True and r["detections"][0]["calibrated_prob"] == 0.98
    assert r["count_intervals"]["probabilities"] == "calibrated"
    assert plain["count_intervals"]["probabilities"] == "raw" and plain["counts"]["possibly_unknown"] == 0


def test_reports_show_calibration_unknown_and_sensitivity(tmp_path) -> None:
    from src.pipeline.report import detections_frame, report_csv, summary_lines
    res = _calibrated_service(tmp_path).analyze(write_image(tmp_path / "a.png"))
    r = res.to_dict()
    det = detections_frame(r)
    assert det.loc[0, "possibly_unknown"] == "yes" and det.loc[1, "possibly_unknown"] == ""
    assert det["calibrated_prob"].tolist() == [0.98] * len(det)
    fields = dict(summary_lines(r))
    assert "Threshold sensitivity" in fields and fields["Possibly unknown"].startswith("1 detections")
    assert "possibly_unknown" in report_csv(r)


def test_missing_calibration_is_a_load_warning(tmp_path) -> None:
    class NoCal(FakePipeline):
        confidence = None
        confidence_warning = "No confidence calibration (models/confidence.npz): off."
    from src.service import AnalysisService
    det, cls = tmp_path / "det.pt", tmp_path / "cls.pt"
    det.write_bytes(b"d"), cls.write_bytes(b"c")
    svc = AnalysisService(load_config(), pipeline=NoCal(), detector_path=det, classifier_path=cls, warmup=False)
    assert "No confidence calibration" in svc.analyze(write_image(tmp_path / "a.png")).warnings[0]
