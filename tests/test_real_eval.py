"""Real-image pilot: manual count parsing, predicted counts and error summaries."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.counting import ORGANISM, REJECT, UNCERTAIN, Detection
from src.pipeline.real_eval import TAG, compare, list_images, load_manual_counts, predicted_counts, unmatched

ORGS = ["copepod", "diatom", "rotifer"]


def det(label, status):
    return Detection(box=(0, 0, 1, 1), det_conf=0.9, label=label, prob=0.9, status=status)


def test_total_only_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("image,total_count\na.jpg,5\nb,\n")
    m, cols, ignored = load_manual_counts(p, ORGS)
    assert cols == [] and ignored == []
    assert list(m.index) == ["a", "b"]                 # extension stripped
    assert m.loc["a", "total_count"] == 5 and np.isnan(m.loc["b", "total_count"])


def test_per_class_and_ignored_columns(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("image, total_count ,copepod,rotifer,comment\na.png,4,1,2,hi\n")
    m, cols, ignored = load_manual_counts(p, ORGS)
    assert cols == ["copepod", "rotifer"] and ignored == ["comment"]
    assert m.loc["a", "rotifer"] == 2


@pytest.mark.parametrize("text", ["image,count\na,1\n", "image,total_count\na,1\na.jpg,2\n"])
def test_bad_csv_rejected(tmp_path, text):
    p = tmp_path / "c.csv"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_manual_counts(p, ORGS)


def test_predicted_counts_keep_uncertain_and_reject_separate():
    dets = [det("copepod", ORGANISM), det("copepod", ORGANISM), det("rotifer", UNCERTAIN),
            det("reject", REJECT), det("diatom", ORGANISM)]
    c = predicted_counts(dets, ORGS)
    assert c["pred_copepod"] == 2 and c["pred_diatom"] == 1 and c["pred_rotifer"] == 0
    assert c["pred_uncertain"] == 1 and c["pred_reject"] == 1
    assert c["pred_total_confident"] == 3 and c["pred_total_organisms_plus_uncertain"] == 4
    assert c["detections"] == 5


def _pred(rows):
    return pd.DataFrame(rows).set_index("image")


def test_compare_totals_only_and_missing_values():
    pred = _pred([{"image": "a", "pred_total_confident": 3, "pred_total_organisms_plus_uncertain": 5},
                  {"image": "b", "pred_total_confident": 1, "pred_total_organisms_plus_uncertain": 1},
                  {"image": "c", "pred_total_confident": 0, "pred_total_organisms_plus_uncertain": 0}])
    manual = pd.DataFrame({"total_count": [4.0, 3.0, np.nan]}, index=pd.Index(["a", "b", "c"], name="image"))
    table, summary = compare(pred, manual, [])
    s = summary.set_index("quantity")
    main = s.loc["total (organisms + uncertain)"]
    assert main["n_images"] == 2                          # NaN manual count excluded
    assert main["MAE"] == pytest.approx(1.5)              # |5-4|, |1-3|
    assert main["mean_signed_error"] == pytest.approx(-0.5)
    assert s.loc["total (confident organisms only)", "mean_signed_error"] == pytest.approx(-1.5)
    assert (table["note"] == TAG).all() and (summary["note"] == TAG).all()


def test_compare_per_class():
    pred = _pred([{"image": "a", "pred_total_confident": 2, "pred_total_organisms_plus_uncertain": 2,
                   "pred_copepod": 2}])
    manual = pd.DataFrame({"total_count": [2.0], "copepod": [3.0]}, index=pd.Index(["a"], name="image"))
    _, summary = compare(pred, manual, ["copepod"])
    row = summary.set_index("quantity").loc["copepod"]
    assert row["MAE"] == 1 and row["mean_signed_error"] == -1


def test_compare_without_manual_counts():
    pred = _pred([{"image": "a", "pred_total_confident": 2, "pred_total_organisms_plus_uncertain": 2}])
    table, summary = compare(pred, None, [])
    assert summary.empty and (table["note"] == TAG).all()


def test_unmatched_and_list_images(tmp_path):
    manual = pd.DataFrame({"total_count": [1, 2]}, index=["a", "z"])
    assert unmatched(["a", "b"], manual) == (["b"], ["z"])
    for n in ("x.jpg", "y.PNG", "notes.txt"):
        (tmp_path / n).write_bytes(b"0")
    assert [p.name for p in list_images(tmp_path)] == ["x.jpg", "y.PNG"]
