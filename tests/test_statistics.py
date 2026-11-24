"""Tests for src/pipeline/statistics.py and its use in the service: intervals and concentrations."""

import itertools
import math

import numpy as np
import pytest

from src.pipeline.counting import Detection
from src.pipeline.statistics import (
    Calibration, classification_intervals, concentration, concentration_block, poisson_binomial_pmf, poisson_ci,
    pooled_concentration, soft_count_interval,
)
from src.service import AnalysisSettings
from tests.fakes import fake_service, write_image


# ---------------- Poisson (Garwood) interval: published reference values
@pytest.mark.parametrize("n, low, high", [(0, 0.0, 3.689), (1, 0.0253, 5.572), (10, 4.795, 18.390),
                                          (100, 81.364, 121.627)])
def test_poisson_ci_matches_reference_values(n, low, high) -> None:
    lo, hi = poisson_ci(n)
    assert lo == pytest.approx(low, abs=2e-3) and hi == pytest.approx(high, abs=2e-2)


def test_poisson_ci_rejects_bad_counts() -> None:
    for n in (-1, 2.5):
        with pytest.raises(ValueError):
            poisson_ci(n)


# ---------------- Poisson-binomial
def test_pmf_matches_brute_force_enumeration() -> None:
    ps = [0.9, 0.2, 0.55, 0.7]
    brute = np.zeros(len(ps) + 1)
    for outcome in itertools.product([0, 1], repeat=len(ps)):
        brute[sum(outcome)] += math.prod(p if o else 1 - p for p, o in zip(ps, outcome))
    assert np.allclose(poisson_binomial_pmf(ps), brute) and poisson_binomial_pmf(ps).sum() == pytest.approx(1)


def test_certain_probabilities_give_an_exact_count() -> None:
    assert soft_count_interval([1.0] * 4 + [0.0] * 3) == {"expected": 4.0, "low": 4, "high": 4}
    assert soft_count_interval([]) == {"expected": 0.0, "low": 0, "high": 0}


def test_interval_holds_at_least_the_level() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        ps = rng.uniform(0, 1, rng.integers(1, 25))
        iv = soft_count_interval(ps)
        pmf = poisson_binomial_pmf(ps)
        assert pmf[iv["low"]: iv["high"] + 1].sum() >= 0.95 - 1e-9
        assert iv["low"] <= iv["expected"] <= iv["high"]


def test_invalid_probability_is_refused() -> None:
    with pytest.raises(ValueError):
        poisson_binomial_pmf([0.5, 1.2])


def test_classification_intervals_sum_to_the_detections() -> None:
    dets = [Detection((0, 0, 1, 1), 0.9, "copepod", 0.6, probs={"copepod": 0.6, "diatom": 0.3, "reject": 0.1}),
            Detection((0, 0, 1, 1), 0.9, "diatom", 0.95, probs={"copepod": 0.05, "diatom": 0.95, "reject": 0.0})]
    iv = classification_intervals(dets, ["copepod", "diatom", "reject"])["per_class"]
    assert sum(v["expected"] for v in iv.values()) == pytest.approx(2)
    assert iv["copepod"] == {"expected": 0.65, "low": 0, "high": 2} and iv["diatom"] == {"expected": 1.25, "low": 0, "high": 2}


# ---------------- calibration and concentration
def test_calibration_from_geometry() -> None:
    # 1000 x 1000 px at 1 um/px, 1000 um deep = 1 mm^3 = 0.001 mL
    assert Calibration(pixel_size_um=1.0, depth_um=1000.0).volume_ml(1000, 1000) == pytest.approx(1e-3)
    assert Calibration(imaged_volume_ml=0.05).volume_ml(10, 10) == 0.05


@pytest.mark.parametrize("kwargs", [{}, {"pixel_size_um": 1.0}, {"imaged_volume_ml": 0.0},
                                    {"imaged_volume_ml": -1}, {"imaged_volume_ml": 1, "depth_um": 5},
                                    {"imaged_volume_ml": 1, "concentration_factor": 0},
                                    {"imaged_volume_ml": float("nan")}])
def test_calibration_rejects_incomplete_or_invalid_input(kwargs) -> None:
    with pytest.raises(ValueError):
        Calibration(**kwargs)


def test_concentration_divides_by_the_original_water_volume() -> None:
    # 10 organisms in 0.1 mL of a sample concentrated 100x: 0.1 x 100 = 10 mL of lake water -> 1 per mL
    c = concentration({"diatom": 10}, volume_ml=0.1, factor=100)["diatom"]
    assert c["per_ml"] == pytest.approx(1.0) and c["per_l"] == pytest.approx(1000)
    lo, hi = poisson_ci(10)
    assert c["per_ml_low"] == pytest.approx(lo / 10) and c["per_ml_high"] == pytest.approx(hi / 10)


def test_zero_count_still_has_an_upper_bound() -> None:
    c = concentration({"rotifer": 0}, volume_ml=1.0)["rotifer"]
    assert c["per_ml"] == 0 and c["per_ml_low"] == 0 and c["per_ml_high"] == pytest.approx(3.689, abs=1e-3)


def test_pooling_narrows_the_interval() -> None:
    calib = Calibration(imaged_volume_ml=0.01)
    one = concentration_block({"diatom": 4, "copepod": 1}, calib, 100, 100)
    pooled = pooled_concentration([{"concentration": one}] * 5 + [{"concentration": None}])
    assert pooled["images"] == 5 and pooled["water_volume_ml"] == pytest.approx(0.05)
    p, s = pooled["per_class"]["diatom"], one["per_class"]["diatom"]
    assert p["per_ml"] == pytest.approx(s["per_ml"])                       # same rate
    assert p["per_ml_high"] - p["per_ml_low"] < s["per_ml_high"] - s["per_ml_low"]
    assert pooled["total"]["count"] == 25
    assert pooled_concentration([{"concentration": None}]) is None


# ---------------- in the service result
def test_result_has_intervals_and_no_concentration_by_default(tmp_path) -> None:
    r = fake_service(tmp_path).analyze(write_image(tmp_path / "a.png")).to_dict()
    iv = r["count_intervals"]["per_class"]
    assert r["concentration"] is None
    assert set(iv) == set(r["counts"]["confirmed"]) | {"reject"}
    assert sum(v["expected"] for v in iv.values()) == pytest.approx(r["counts"]["total_detections"], abs=0.01)


def test_result_has_concentration_with_calibration(tmp_path) -> None:
    svc = fake_service(tmp_path)
    s = AnalysisSettings(calibration=Calibration(pixel_size_um=2.0, depth_um=100.0, concentration_factor=10))
    res = svc.analyze(write_image(tmp_path / "a.png", h=120, w=160), s)
    r = res.to_dict()
    conc = r["concentration"]
    vol = 160 * 120 * 4.0 * 100.0 / 1e12
    assert conc["imaged_volume_ml"] == pytest.approx(vol) and conc["water_volume_ml"] == pytest.approx(10 * vol)
    assert conc["total"]["count"] == r["counts"]["confirmed_total"]
    for c, v in conc["per_class"].items():
        assert v["count"] == r["counts"]["confirmed"][c]
    assert any("concentration interval is wide" in w for w in r["warnings"])


# ---------------- reports, API, CLI
def test_reports_carry_intervals_and_concentration(tmp_path) -> None:
    from src.pipeline.report import concentration_frame, counts_frame, report_csv, report_html, report_pdf
    from tests.test_report import _csv_sections
    s = AnalysisSettings(calibration=Calibration(imaged_volume_ml=0.002, concentration_factor=50))
    res = fake_service(tmp_path).analyze(write_image(tmp_path / "a.png"), s)
    r = res.to_dict()
    t = counts_frame(r).set_index("class")
    for c, v in r["count_intervals"]["per_class"].items():
        row = t.loc["rejected" if c == "reject" else c]
        assert (row["expected"], row["interval_low"], row["interval_high"]) == (v["expected"], v["low"], v["high"])
    conc = concentration_frame(r).set_index("class")
    assert conc.loc["ALL organisms", "count"] == r["counts"]["confirmed_total"]
    # 0.002 mL x 50 = 0.1 mL of water, so per litre = count x 10 000
    assert conc.loc["ALL organisms", "per_l"] == pytest.approx(r["counts"]["confirmed_total"] * 1e4)
    sections = _csv_sections(report_csv(r))
    key = next(k for k in sections if k.startswith("concentration"))
    assert list(sections[key]["class"])[-1] == "ALL organisms"
    assert "Concentration (per litre" in report_html(r, res.annotated) and report_pdf(r, res.annotated)[:4] == b"%PDF"


def test_old_results_without_intervals_still_render(tmp_path) -> None:
    from src.pipeline.report import concentration_frame, counts_frame
    r = fake_service(tmp_path).analyze(write_image(tmp_path / "a.png")).to_dict()
    r.pop("count_intervals"), r.pop("concentration")
    assert counts_frame(r)["expected"].isna().all() and concentration_frame(r) is None


def test_api_calibration_fields(tmp_path) -> None:
    from fastapi.testclient import TestClient
    from api.main import create_app
    from tests.test_api import _post
    app = create_app(service=fake_service(tmp_path), history_root=tmp_path / "hist", load_models=False,
                     frontend_dist=tmp_path / "no_dist")
    with TestClient(app) as c:
        ok = _post(c, pixel_size_um=2.0, depth_um=100.0)
        assert ok.status_code == 201 and ok.json()["result"]["concentration"]["calibration"] == \
            {"pixel_size_um": 2.0, "depth_um": 100.0, "concentration_factor": 1.0}
        assert _post(c).json()["result"]["concentration"] is None
        for bad in ({"pixel_size_um": 2.0}, {"imaged_volume_ml": 0.1, "depth_um": 5}, {"imaged_volume_ml": -1},
                    {"concentration_factor": 10}):
            r = _post(c, **bad)
            assert r.status_code == 422 and r.json()["error"]["code"] == "bad_request", bad


def test_cli_pooled_concentration(tmp_path, capsys) -> None:
    import json
    from scripts.analyze import main
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        write_image(src / f"f{i}.png")
    out = tmp_path / "out"
    assert main([str(src), "--out", str(out), "--no-history", "--volume-ml", "0.01"], service=fake_service(tmp_path)) == 0
    pooled = json.loads(next(out.glob("pooled_concentration_*.json")).read_text())
    assert pooled["images"] == 3 and pooled["water_volume_ml"] == pytest.approx(0.03)
    assert pooled["total"]["count"] == 6                                    # 2 confirmed per fake image
    assert "pooled over 3 image(s)" in capsys.readouterr().out
    assert main([str(src), "--out", str(out), "--no-history", "--depth-um", "5"], service=fake_service(tmp_path)) == 2
