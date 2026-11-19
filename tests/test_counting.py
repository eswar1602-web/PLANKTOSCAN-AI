"""Tests for status assignment and counting (uncertain/reject never silently dropped)."""

import pytest

from src.pipeline.counting import (
    REJECT, UNCERTAIN, Detection, assign_status, count_detections, counts_table,
)

ORGS = ["copepod", "diatom", "nauplius"]


def _det(label: str, prob: float, thr: float = 0.6) -> Detection:
    return Detection(box=(0, 0, 10, 10), det_conf=0.9, label=label, prob=prob,
                     status=assign_status(label, prob, thr, "reject"))


@pytest.mark.parametrize("label, prob, expected", [
    ("copepod", 0.95, "organism"),
    ("copepod", 0.59, UNCERTAIN),          # below threshold -> uncertain
    ("reject", 0.95, REJECT),              # confident reject
    ("reject", 0.30, UNCERTAIN),           # low confidence wins over the reject label
    ("diatom", 0.60, "organism"),          # threshold is inclusive
])
def test_assign_status(label, prob, expected) -> None:
    assert assign_status(label, prob, 0.6, "reject") == expected


def test_counts_include_every_detection_once() -> None:
    dets = [_det("copepod", 0.9), _det("copepod", 0.8), _det("diatom", 0.7),
            _det("nauplius", 0.2), _det("reject", 0.99), _det("reject", 0.1)]
    c = count_detections(dets, ORGS)
    assert c == {"copepod": 2, "diatom": 1, "nauplius": 0, UNCERTAIN: 2, REJECT: 1, "total": 6}


def test_reject_and_uncertain_excluded_from_organism_counts() -> None:
    dets = [_det("reject", 0.99), _det("copepod", 0.1)]
    c = count_detections(dets, ORGS)
    assert sum(c[o] for o in ORGS) == 0 and c[REJECT] == 1 and c[UNCERTAIN] == 1


def test_empty_image() -> None:
    c = count_detections([], ORGS)
    assert c["total"] == 0 and all(v == 0 for v in c.values())


def test_counts_table_rows_and_order() -> None:
    dets = [_det("copepod", 0.9), _det("diatom", 0.3), _det("reject", 0.95)]
    t = counts_table(dets, ORGS)
    assert list(t["class"]) == ORGS + [UNCERTAIN, REJECT]
    assert t.set_index("class")["count"].to_dict() == {"copepod": 1, "diatom": 0, "nauplius": 0,
                                                       UNCERTAIN: 1, REJECT: 1}
    assert t["count"].sum() == len(dets)


# ---------------- estimated_counts (argmax per class, uncertain kept separate) ----------------
from src.pipeline.counting import estimated_counts  # noqa: E402

ALL = ORGS + ["reject"]


def test_estimated_counts_example() -> None:
    dets = [_det("copepod", 0.9), _det("copepod", 0.3), _det("diatom", 0.2), _det("reject", 0.95),
            _det("reject", 0.1)]
    e = estimated_counts(dets, ORGS)
    assert {k: e[k] for k in ALL} == {"copepod": 2, "diatom": 1, "nauplius": 0, "reject": 2}
    assert e["from_uncertain"] == {"copepod": 1, "diatom": 1, "nauplius": 0, "reject": 1}
    assert e["total"] == 5


def test_estimated_counts_empty() -> None:
    e = estimated_counts([], ORGS)
    assert e["total"] == 0 and all(e[k] == 0 for k in ALL) and sum(e["from_uncertain"].values()) == 0


@pytest.mark.parametrize("seed", range(25))
def test_estimated_and_strict_counts_always_match_the_box_list(seed) -> None:
    """Random detection lists: both count dicts are fully determined by, and add up to, the final boxes."""
    import random
    rng = random.Random(seed)
    dets = [_det(rng.choice(ALL), rng.random(), thr=0.6) for _ in range(rng.randint(0, 40))]
    s, e = count_detections(dets, ORGS), estimated_counts(dets, ORGS)
    assert sum(e[k] for k in ALL) == e["total"] == s["total"] == len(dets)
    assert sum(e["from_uncertain"].values()) == s[UNCERTAIN]
    for c in ORGS:   # estimate = confident count + uncertain items whose argmax is c
        assert e[c] == s[c] + e["from_uncertain"][c] == sum(d.label == c for d in dets)
    assert e["reject"] == s[REJECT] + e["from_uncertain"]["reject"]
    assert all(e[k] >= s.get(k, 0) for k in ORGS)


def test_estimated_counts_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        estimated_counts([_det("fish", 0.9)], ORGS)


def test_frame_record_and_summary() -> None:
    from src.evaluation import frame_record, summarize_frames
    objs = [{"box": [0, 0, 10, 10], "target": "copepod"}, {"box": [20, 20, 30, 30], "target": "diatom"},
            {"box": [40, 40, 50, 50], "target": "copepod"}]
    dets = [Detection(box=(0, 0, 10, 10), det_conf=0.9, label="copepod", prob=0.9, status="organism"),
            Detection(box=(20, 20, 30, 30), det_conf=0.8, label="diatom", prob=0.3, status=UNCERTAIN),
            Detection(box=(70, 70, 80, 80), det_conf=0.7, label="nauplius", prob=0.9, status="organism")]
    r = frame_record(dets, objs, ORGS, "reject", 0.5)
    assert r["tp"] == 2 and r["matched_uncertain"] == 1 and r["matched_correct"] == 1
    assert r["strict"] == {"copepod": 1, "diatom": 0, "nauplius": 1, "reject": 0}
    assert r["estimated"] == {"copepod": 1, "diatom": 1, "nauplius": 1, "reject": 0}
    s, strict, est = summarize_frames([r], ORGS, "reject")
    assert s["strict_count_signed_error"] == -1 and s["estimated_count_signed_error"] == 0
    assert s["matched_uncertain_pct"] == 50 and abs(s["det_f1"] - 2 / 3) < 1e-9
    assert strict.loc["copepod", "mean_signed_error"] == -1 and est.loc["diatom", "MAE"] == 0
