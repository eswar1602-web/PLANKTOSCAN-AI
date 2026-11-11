"""Tests for raw-class -> project-class mapping and open-set separation."""

import pandas as pd
import pytest

from src.common.config import load_config
from src.data.mapping import apply_mapping, invert_mapping, validate_mapping

MAPPING = {"copepod": ["cyclops", "eudiaptomus"], "nauplius": ["nauplius"], "reject": ["dirt"]}
OPEN = ["fish"]
RAW = ["cyclops", "eudiaptomus", "nauplius", "dirt", "fish"]


def _inv() -> pd.DataFrame:
    return pd.DataFrame({"raw_class": RAW + ["cyclops"], "path": [f"{i}.jpeg" for i in range(6)]})


def test_invert_mapping() -> None:
    assert invert_mapping(MAPPING) == {
        "cyclops": "copepod", "eudiaptomus": "copepod", "nauplius": "nauplius", "dirt": "reject"
    }


def test_raw_class_in_two_targets_rejected() -> None:
    with pytest.raises(ValueError, match="mapped to both"):
        invert_mapping({"a": ["x"], "b": ["x"]})


@pytest.mark.parametrize("mapping, open_set, match", [
    (MAPPING, OPEN + ["dirt"], "both mapped and open-set"),
    ({**MAPPING, "extra": ["not_in_data"]}, OPEN, "not in the data"),
    (MAPPING, [], "neither mapped nor open-set"),
])
def test_validate_mapping_errors(mapping, open_set, match) -> None:
    with pytest.raises(ValueError, match=match):
        validate_mapping(mapping, open_set, RAW)


def test_apply_mapping_splits_closed_and_open() -> None:
    closed, opened = apply_mapping(_inv(), MAPPING, OPEN)
    assert len(closed) + len(opened) == 6
    assert set(closed["target"]) == {"copepod", "nauplius", "reject"}
    assert (closed.loc[closed["raw_class"] == "cyclops", "target"] == "copepod").all()
    assert list(opened["raw_class"]) == ["fish"]
    assert "target" not in opened.columns


def test_project_config_mapping_is_consistent() -> None:
    """The real config: 8 classes incl. the required three, 25 mapped + 10 open-set raw classes."""
    data = load_config()["data"]
    cm = data["class_mapping"]
    assert len(cm) == 8 and {"copepod", "diatom", "nauplius", "reject"} <= set(cm)
    assert data["reject_class"] in cm
    mapped = invert_mapping(cm)
    assert len(mapped) == 25 and len(data["open_set"]) == 10
    assert not set(mapped) & set(data["open_set"])
