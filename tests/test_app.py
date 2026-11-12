"""Headless smoke test of the Streamlit app (streamlit.testing AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
MODELS_READY = (ROOT / "models" / "detector_best.pt").exists() and (ROOT / "models" / "classifier_best.pt").exists()
EXAMPLES = (ROOT / "data" / "synthetic" / "images" / "val").exists()


def test_app_starts_and_shows_domain_warning() -> None:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert at.title[0].value == "PlanktoScan AI"
    assert any("NOT validated" in w.value for w in at.warning)


@pytest.mark.skipif(not (MODELS_READY and EXAMPLES), reason="needs trained models and synthetic val frames")
def test_app_runs_pipeline_on_example() -> None:
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.checkbox[0].check().run()
    assert not at.exception
    assert len(at.dataframe) == 2                         # counts + detections tables
    counts = at.dataframe[0].value
    assert {"uncertain", "reject"} <= set(counts["class"])
    assert len(at.sidebar.slider) == 2
