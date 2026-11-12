"""Status assignment and counting. Uncertain and reject items are counted separately, never dropped."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

ORGANISM, UNCERTAIN, REJECT = "organism", "uncertain", "reject"


@dataclass
class Detection:
    """One detected object and its classification."""
    box: tuple[float, float, float, float]          # x0, y0, x1, y1 in image pixels
    det_conf: float                                 # detector confidence
    label: str                                      # classifier argmax class
    prob: float                                     # classifier max softmax probability
    status: str = ORGANISM                          # organism | uncertain | reject
    probs: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d.update(x0=self.box[0], y0=self.box[1], x1=self.box[2], y1=self.box[3])
        d.pop("box"); d.pop("probs")
        return d


def assign_status(label: str, prob: float, threshold: float, reject_class: str) -> str:
    """Low confidence -> uncertain (whatever the label); else reject class -> reject; else organism."""
    if prob < threshold:
        return UNCERTAIN
    if label == reject_class:
        return REJECT
    return ORGANISM


def count_detections(dets: list[Detection], organism_classes: list[str]) -> dict[str, int]:
    """Per-organism-class counts (confident organisms only) plus 'uncertain', 'reject', 'total'."""
    counts = {c: 0 for c in organism_classes}
    counts[UNCERTAIN] = counts[REJECT] = 0
    for d in dets:
        if d.status == ORGANISM:
            counts[d.label] += 1
        else:
            counts[d.status] += 1
    counts["total"] = len(dets)
    assert sum(v for k, v in counts.items() if k != "total") == len(dets), "every detection counted once"
    return counts


def counts_table(dets: list[Detection], organism_classes: list[str]) -> pd.DataFrame:
    """Table for display/report: one row per organism class, then 'uncertain' and 'reject' rows."""
    counts = count_detections(dets, organism_classes)
    rows = []
    for name in organism_classes + [UNCERTAIN, REJECT]:
        if name in organism_classes:
            sel = [d for d in dets if d.status == ORGANISM and d.label == name]
            kind = "organism"
        else:
            sel = [d for d in dets if d.status == name]
            kind = "flagged, not counted" if name == UNCERTAIN else "rejected, not counted"
        mean_p = round(sum(d.prob for d in sel) / len(sel), 3) if sel else None
        rows.append({"class": name, "count": counts[name], "type": kind, "mean_classifier_prob": mean_p})
    return pd.DataFrame(rows)
