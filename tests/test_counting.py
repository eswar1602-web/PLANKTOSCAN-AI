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
