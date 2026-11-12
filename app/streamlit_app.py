"""PlanktoScan AI - Streamlit UI.

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from src.common.config import load_config, resolve_path
from src.pipeline.counting import counts_table
from src.pipeline.report import DOMAIN_WARNING, detections_frame, to_csv, to_html, to_pdf

st.set_page_config(page_title="PlanktoScan AI", layout="wide")
st.title("PlanktoScan AI")
st.caption("Detect, classify and count plankton: YOLOv8n detector + EfficientNet-B0 classifier "
           "(ImageNet/COCO-pretrained, fine-tuned on ZooLake-derived data).")
st.warning(DOMAIN_WARNING)

cfg = load_config()
models = resolve_path(cfg, "models")
if not (models / "detector_best.pt").exists() or not (models / "classifier_best.pt").exists():
    st.error("Trained models not found in models/. Train them first (see README).")
    st.stop()


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline():
    from src.pipeline.pipeline import Pipeline
    return Pipeline(load_config())


pipe = get_pipeline()

with st.sidebar:
    st.header("Thresholds")
    det_conf = st.slider("Detection confidence", 0.05, 0.95, float(round(pipe.det_conf, 2)), 0.01,
                         help="Chosen on synthetic VAL frames (max detection F1).")
    unc = st.slider("Uncertainty threshold (classifier max probability)", 0.0, 1.0,
                    float(round(pipe.uncertainty, 2)), 0.01,
                    help="Below this the item is shown as 'uncertain' and not counted as an organism. "
                         "Default chosen on the VAL split.")
    st.caption(f"NMS IoU (tuned on val): {pipe.nms_iou}")

uploaded = st.file_uploader("Upload a microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
example_dir = resolve_path(cfg, "synthetic_data") / "images" / "val"
use_example = st.checkbox("Use an example synthetic VAL frame", value=False,
                          disabled=not example_dir.exists(),
                          help="A generated frame from validation crops (not real microscope data).")

img, name = None, None
if uploaded is not None:
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read this image file.")
    else:
        img, name = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), uploaded.name
elif use_example:
    path = sorted(example_dir.glob("*.jpg"))[0]
    img, name = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), path.name

if img is not None:
    from src.pipeline.pipeline import annotate
    with st.spinner("Detecting and classifying..."):
        dets = pipe.run(img, det_conf=det_conf, uncertainty=unc)
    annotated = annotate(img, dets)
    table = counts_table(dets, pipe.organisms)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(annotated, caption=f"{name}: blue = counted organism, orange = uncertain, grey = reject",
                 use_container_width=True)
    with col2:
        st.subheader("Counts")
        st.dataframe(table, hide_index=True, use_container_width=True)
        n_org = int(table[table["type"] == "organism"]["count"].sum())
        st.metric("Organisms counted", n_org,
                  help=f"{len(dets)} detections in total, including uncertain and reject items")
        stem = Path(name).stem
        st.download_button("Download CSV", to_csv(dets, pipe.organisms), f"{stem}_planktoscan.csv", "text/csv")
        st.download_button("Download HTML report", to_html(dets, pipe.organisms, annotated, name, det_conf, unc),
                           f"{stem}_planktoscan.html", "text/html")
        st.download_button("Download PDF report", to_pdf(dets, pipe.organisms, annotated, name, det_conf, unc),
                           f"{stem}_planktoscan.pdf", "application/pdf")
    st.subheader("All detections")
    st.dataframe(detections_frame(dets), hide_index=True, use_container_width=True)
