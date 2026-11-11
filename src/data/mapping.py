"""Map raw ZooLake classes to project classes and separate the open-set images."""

from __future__ import annotations

import pandas as pd

OPEN_SET = "__open_set__"


def invert_mapping(class_mapping: dict[str, list[str]]) -> dict[str, str]:
    """Turn {target: [raw, ...]} into {raw: target}; raise if a raw class appears twice."""
    raw_to_target: dict[str, str] = {}
    for target, raws in class_mapping.items():
        for raw in raws:
            if raw in raw_to_target:
                raise ValueError(f"raw class '{raw}' mapped to both '{raw_to_target[raw]}' and '{target}'")
            raw_to_target[raw] = target
    return raw_to_target


def validate_mapping(
    class_mapping: dict[str, list[str]], open_set: list[str], raw_classes: list[str]
) -> None:
    """Every raw class must be either mapped or open-set (not both, none unknown, none forgotten)."""
    mapped = set(invert_mapping(class_mapping))
    opened = set(open_set)
    raw = set(raw_classes)
    if both := mapped & opened:
        raise ValueError(f"classes both mapped and open-set: {sorted(both)}")
    if unknown := (mapped | opened) - raw:
        raise ValueError(f"config names raw classes not in the data: {sorted(unknown)}")
    if forgotten := raw - mapped - opened:
        raise ValueError(f"raw classes neither mapped nor open-set: {sorted(forgotten)}")


def apply_mapping(
    inv: pd.DataFrame, class_mapping: dict[str, list[str]], open_set: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (closed_set, open_set) frames; closed_set gets a 'target' column."""
    validate_mapping(class_mapping, open_set, sorted(inv["raw_class"].unique()))
    raw_to_target = invert_mapping(class_mapping)
    df = inv.copy()
    df["target"] = df["raw_class"].map(raw_to_target).fillna(OPEN_SET)
    closed = df[df["target"] != OPEN_SET].reset_index(drop=True)
    opened = df[df["target"] == OPEN_SET].drop(columns="target").reset_index(drop=True)
    return closed, opened
