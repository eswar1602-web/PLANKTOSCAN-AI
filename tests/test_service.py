"""Tests for src/service.py: validation, typed errors, the canonical result, timings, warnings, outputs."""

import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from src.common.config import load_config
from src.service import (
    NO_ORGANISMS, STATUS_EMPTY, STATUS_OK, AnalysisSettings, ImageTooLargeError, ImageTooSmallError, InferenceError,
    InvalidImageError, ModelLoadError, UnsupportedFormatError, AnalysisService, load_saved_result, read_image_file,
    safe_stem, save_outputs, structured_counts, validate_bytes,
)
from tests.fakes import FakePipeline, fake_service, write_image

SCFG = load_config()["service"]


def _png(arr: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    return enc.tobytes()


# ---------------- validation
def test_valid_formats_decode_to_uint8_rgb(tmp_path) -> None:
    for name, ch, dt in [("a.png", 3, np.uint8), ("b.jpg", 3, np.uint8), ("c.tif", 1, np.uint8),
                         ("d.bmp", 3, np.uint8), ("e.png", 1, np.uint16), ("f.png", 4, np.uint8)]:
        img, data = read_image_file(write_image(tmp_path / name, channels=ch, dtype=dt), SCFG)
        assert img.shape == (120, 160, 3) and img.dtype == np.uint8 and len(data) > 0, name


def test_bgr_is_converted_to_rgb() -> None:
    bgr = np.zeros((40, 40, 3), np.uint8)
    bgr[..., 0] = 255                                    # pure blue in OpenCV's BGR order
    rgb = validate_bytes(_png(bgr), "x.png", SCFG)
    assert rgb[0, 0].tolist() == [0, 0, 255]


@pytest.mark.parametrize("name, data, err", [
    ("x.gif", b"GIF89a....", UnsupportedFormatError),
    ("x", b"\x89PNG", UnsupportedFormatError),
    ("x.png", b"", InvalidImageError),
    ("x.png", b"this is not an image at all", InvalidImageError),
    ("x.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 50, InvalidImageError),
])
def test_bad_inputs_raise_typed_errors(name, data, err) -> None:
    with pytest.raises(err):
        validate_bytes(data, name, SCFG)


def test_truncated_jpeg_is_invalid() -> None:
    ok, enc = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8))
    data = enc.tobytes()
    with pytest.raises(InvalidImageError):
        validate_bytes(data[: len(data) // 3], "t.jpg", SCFG)


def test_gif_content_renamed_to_png_is_unsupported() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="GIF")
    with pytest.raises(UnsupportedFormatError):
        validate_bytes(buf.getvalue(), "renamed.png", SCFG)


def test_size_limits() -> None:
    with pytest.raises(ImageTooSmallError):
        validate_bytes(_png(np.zeros((20, 200, 3), np.uint8)), "s.png", SCFG)
    with pytest.raises(ImageTooLargeError):                       # pixel limit, read from the header
        validate_bytes(_png(np.zeros((1200, 1000, 3), np.uint8)), "l.png", {**SCFG, "max_megapixels": 1})
    with pytest.raises(ImageTooLargeError):                       # file-size limit
        validate_bytes(b"\x89PNG" + b"0" * 2_000_000, "l.png", {**SCFG, "max_file_mb": 1})


def test_missing_file_and_oversized_file(tmp_path) -> None:
    with pytest.raises(InvalidImageError, match="not found"):
        read_image_file(tmp_path / "nope.png", SCFG)
    p = write_image(tmp_path / "big.png")
    with pytest.raises(ImageTooLargeError):
        read_image_file(p, {**SCFG, "max_file_mb": 1e-6})


def test_all_errors_share_base_class_and_have_codes() -> None:
    from src.service import AnalysisError
    codes = set()
    for cls in (InvalidImageError, UnsupportedFormatError, ImageTooSmallError, ImageTooLargeError, ModelLoadError,
                InferenceError):
        assert issubclass(cls, AnalysisError)
        codes.add(cls.code)
    assert len(codes) == 6


# ---------------- analysis
def test_analyze_returns_complete_consistent_result(tmp_path) -> None:
    svc = fake_service(tmp_path)
    r = svc.analyze(write_image(tmp_path / "frame.png"))
    assert r.status == STATUS_OK and r.filename == "frame.png" and r.source_path.endswith("frame.png")
    assert r.original.shape == r.annotated.shape == (120, 160, 3) and r.enhanced.ndim == 3
    t = svc.pipe.uncertainty                    # default (VAL-chosen) threshold used
    assert svc.pipe.calls == [(svc.pipe.det_conf, t)]
    # preds: copepod .97, diatom .95 confirmed; rotifer .5, dinobryon .3 uncertain; reject .99 rejected
    c, e = r.counts, r.estimated_counts
    assert c["confirmed"]["copepod"] == 1 and c["confirmed"]["diatom"] == 1 and c["confirmed_total"] == 2
    assert c["uncertain"] == 2 and c["rejected"] == 1 and c["total_detections"] == 5
    assert e["per_class"]["rotifer"] == 1 and e["per_class"]["dinobryon"] == 1 and e["organisms_total"] == 4
    assert e["from_uncertain"]["rotifer"] == 1 and e["reject"] == 1
    assert {"load_and_validate", "detection", "classification", "counting", "annotation",
            "enhancement_preview", "total"} <= set(r.timings_ms)
    assert r.timings_ms["total"] >= sum(v for k, v in r.timings_ms.items() if k != "total") - 1e-6
    # 2 of 5 uncertain = 40% > high_uncertain_warn_fraction (30%): exactly that one warning
    assert r.warnings == ["2 of 5 detections are uncertain: the image may be outside the training domain."]
    assert "NOT validated" in r.metadata["disclaimer"]
    json.dumps(r.to_dict())                     # canonical result is JSON-serialisable
    assert [d["id"] for d in r.to_dict()["detections"]] == [1, 2, 3, 4, 5]


def test_zero_detections_is_a_successful_result_with_warning(tmp_path) -> None:
    r = fake_service(tmp_path, preds=[]).analyze(write_image(tmp_path / "empty.png"))
    assert r.status == STATUS_EMPTY and r.message == NO_ORGANISMS
    assert any(NO_ORGANISMS in w for w in r.warnings)
    assert r.counts["total_detections"] == 0 and r.estimated_counts["total"] == 0
    folder, files = save_outputs(r, tmp_path / "out")          # reports still render
    assert set(files) >= {"json", "csv", "html", "pdf", "annotated", "enhanced"}


def test_inference_failure_becomes_inference_error(tmp_path) -> None:
    svc = fake_service(tmp_path, error=RuntimeError("CUDA error: out of memory"))
    with pytest.raises(InferenceError, match="out of GPU memory"):
        svc.analyze(write_image(tmp_path / "x.png"))
    svc = fake_service(tmp_path, error=KeyError("boom"))
    with pytest.raises(InferenceError):
        svc.analyze(write_image(tmp_path / "y.png"))


def test_missing_weights_raise_model_load_error(tmp_path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        AnalysisService(load_config(), detector_path=tmp_path / "no_det.pt", classifier_path=tmp_path / "no_cls.pt")


def test_corrupt_weights_raise_model_load_error(tmp_path) -> None:
    (tmp_path / "d.pt").write_bytes(b"garbage"), (tmp_path / "c.pt").write_bytes(b"garbage")
    with pytest.raises(ModelLoadError, match="Could not load"):
        AnalysisService(load_config(), detector_path=tmp_path / "d.pt", classifier_path=tmp_path / "c.pt")


def test_warnings_for_non_default_thresholds_high_uncertainty_large_image_and_wrong_run(tmp_path) -> None:
    svc = fake_service(tmp_path, large_side_warn_px=100)
    r = svc.analyze(write_image(tmp_path / "x.png"), AnalysisSettings(det_conf=0.3, uncertainty=0.99))
    text = " ".join(r.warnings)
    assert "Non-default thresholds" in text and "Large image" in text and "uncertain" in text
    assert svc.pipe.calls[-1] == (0.3, 0.99) and r.metadata["thresholds"]["uncertainty"] == 0.99
    r2 = fake_service(tmp_path, run_id="some_other_run").analyze(write_image(tmp_path / "y.png"))
    assert any("chosen for classifier run" in w for w in r2.warnings)


def test_settings_out_of_range(tmp_path) -> None:
    with pytest.raises(ValueError):
        fake_service(tmp_path).analyze(write_image(tmp_path / "x.png"), AnalysisSettings(uncertainty=1.5))


def test_analyze_bytes_matches_analyze(tmp_path) -> None:
    p = write_image(tmp_path / "x.png")
    svc = fake_service(tmp_path)
    a, b = svc.analyze(p), svc.analyze_bytes(p.read_bytes(), "x.png")
    assert a.counts == b.counts and np.array_equal(a.original, b.original) and b.source_path is None
    assert a.analysis_id != b.analysis_id


def test_structured_counts_agree_with_box_list() -> None:
    dets = FakePipeline().run(np.zeros((10, 10, 3), np.uint8))
    c, e = structured_counts(dets, FakePipeline().organisms, "reject")
    assert c["confirmed_total"] + c["uncertain"] + c["rejected"] == len(dets) == e["total"]


# ---------------- outputs
def test_save_outputs_unique_folders_and_consistent_reports(tmp_path) -> None:
    svc = fake_service(tmp_path)
    p = write_image(tmp_path / "frame one.png")
    r1, r2 = svc.analyze(p), svc.analyze(p)
    f1, files1 = save_outputs(r1, tmp_path / "out", original_bytes=p.read_bytes())
    f2, _ = save_outputs(r2, tmp_path / "out", formats=("csv",))
    assert f1 != f2 and f1.parent == f2.parent and "frame_one" in f1.name
    assert (f2 / "report.json").exists() and not (f2 / "report.pdf").exists()   # json is always written
    assert all((f1 / n).exists() for n in ("annotated.png", "enhanced.png", "original.png", "report.csv",
                                           "report.json", "report.html", "report.pdf"))
    saved = load_saved_result(f1)
    assert saved == json.loads(json.dumps(r1.to_dict()))
    with pytest.raises(FileExistsError):                     # never overwrite an existing analysis folder
        save_outputs(r1, tmp_path / "out")
    with pytest.raises(ValueError):
        save_outputs(r2, tmp_path / "out2", formats=("docx",))


def test_safe_stem() -> None:
    assert safe_stem("a b/c?.png") == "c" and safe_stem("../../x y.jpg") == "x_y" and safe_stem("....png") == "image"
    assert len(safe_stem("x" * 300 + ".png")) == 60
