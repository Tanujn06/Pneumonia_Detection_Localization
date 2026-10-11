"""
app.py - Streamlit demo: pneumonia probability + Grad-CAM localization
Run from the repository root:
    streamlit run rc/app/app.py
Author: Member 3
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gradcam import load_model, load_image, explain_multi, overlay_heatmap, DEFAULT_CHECKPOINT, TARGET_LAYERS  # noqa: E402
from localization import (localize, location_prior, parse_config, load_gt_boxes, mask_iou,  # noqa: E402
                          draw_boxes, BASELINE_CONFIG, DEFAULT_METADATA_DIR)


st.set_page_config(page_title="Pneumonia Detection & Localization", layout="wide")


@st.cache_resource
def get_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(checkpoint_path, device), device


@st.cache_data
def get_annotations(annotations_csv):
    return load_gt_boxes(annotations_csv)


@st.cache_data
def get_prior(annotations_csv, train_csv):
    """Train-split location prior (needs all_annotations.csv and train.csv)."""
    return location_prior(pd.read_csv(train_csv)["patientId"], load_gt_boxes(annotations_csv))


def read_upload(uploaded):
    """Write the upload to a temp file so DICOMs go through the same read_xray() as training."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        return load_image(tmp_path)
    finally:
        os.remove(tmp_path)


st.title("Pneumonia Detection & Localization")
st.caption("DenseNet121 classifier (RSNA Pneumonia Detection Challenge) with Grad-CAM heatmaps "
           "and approximate bounding boxes.")
st.warning("Educational project only — **not a clinical diagnosis**. Grad-CAM boxes are approximate "
           "visual explanations, not a trained detector.")

with st.sidebar:
    st.header("Settings")
    checkpoint_path = st.text_input("Checkpoint path",
                                    os.environ.get("PNEUMONIA_CHECKPOINT", DEFAULT_CHECKPOINT))
    annotations_csv = st.text_input("all_annotations.csv (optional, shows ground truth)",
                                    os.environ.get("PNEUMONIA_ANNOTATIONS",
                                                   f"{DEFAULT_METADATA_DIR}/all_annotations.csv"))
    decision_threshold = st.slider("Decision threshold (probability)", 0.05, 0.95, 0.5, 0.05)
    cam_threshold = st.slider("Heatmap threshold for boxes", 0.1, 0.9,
                              float(os.environ.get("PNEUMONIA_CAM_THRESHOLD", 0.5)), 0.05,
                              help="Fraction of the heatmap maximum. Tuned on val.csv by localization.py.")
    max_boxes = st.slider("Max boxes", 1, 4, 3)

    st.subheader("Localization method")
    default = parse_config(os.environ.get("PNEUMONIA_LOC_CONFIG", BASELINE_CONFIG))
    layer = st.selectbox("Grad-CAM layer", TARGET_LAYERS, index=TARGET_LAYERS.index(default["layer"]),
                         help="norm5 = 7x7 heatmap, denseblock3 = 14x14 (finer)")
    use_tta = st.checkbox("Flip test-time augmentation", default["tta"])
    use_split = st.checkbox("One box per lung (split at midline)", default["split"])
    train_csv = str(Path(annotations_csv).with_name("train.csv")) if annotations_csv else ""
    prior_available = bool(train_csv) and os.path.exists(train_csv) and os.path.exists(annotations_csv)
    use_prior = st.checkbox("Location prior (from train boxes)", default["prior"] and prior_available,
                            disabled=not prior_available,
                            help="Needs train.csv next to all_annotations.csv")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)

try:
    model, device = get_model(checkpoint_path)
except FileNotFoundError as e:
    st.error(f"{e}\n\nSet the checkpoint path in the sidebar (or env var PNEUMONIA_CHECKPOINT).")
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray", type=["dcm", "png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a DICOM (.dcm) from the RSNA dataset, or a PNG/JPG chest X-ray.")
    st.stop()

with st.spinner("Running model and Grad-CAM..."):
    image = read_upload(uploaded)
    config = "+".join([layer] + [o for o, on in (("tta", use_tta), ("prior", use_prior), ("split", use_split)) if on])
    cams, prob = explain_multi(model, image, device, (layer,), use_tta)
    prior = get_prior(annotations_csv, train_csv) if use_prior else None
    cam, boxes = localize(cams, config, cam_threshold, prior, max_boxes=max_boxes)

patient_id = Path(uploaded.name).stem
gt_boxes = []
if annotations_csv and os.path.exists(annotations_csv):
    gt_boxes = get_annotations(annotations_csv).get(patient_id, [])

is_positive = prob >= decision_threshold
col_metric, col_info = st.columns([1, 2])
col_metric.metric("P(pneumonia)", f"{prob:.1%}")
col_info.markdown(
    f"**Model output:** {'Pneumonia suspected' if is_positive else 'No pneumonia detected'} "
    f"(threshold {decision_threshold:.2f})  \n"
    f"**Image:** `{uploaded.name}` — {image.size[0]}×{image.size[1]} px  \n"
    f"**Localization:** `{config}`, heatmap threshold {cam_threshold:.2f}"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Input")
    st.image(image, width="stretch")
with col2:
    st.subheader("Grad-CAM heatmap")
    st.image(overlay_heatmap(image, cam, alpha=alpha), width="stretch")
with col3:
    st.subheader("Approximate boxes")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay_heatmap(image, cam, alpha=alpha))
    if gt_boxes:
        draw_boxes(ax, gt_boxes, "lime", "Ground truth")
    draw_boxes(ax, boxes, "red", "Grad-CAM box")
    ax.axis("off")
    if boxes or gt_boxes:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if not is_positive:
    st.caption("The classifier is below the decision threshold, so the heatmap shows what the model "
               "attended to, not a confirmed finding.")

if boxes:
    st.markdown("**Predicted boxes** (original-image pixels, RSNA x/y/width/height format)")
    st.dataframe(pd.DataFrame(boxes).round(1), hide_index=True)
else:
    st.caption("No heatmap region passed the threshold / minimum-area filter.")

if gt_boxes:
    st.markdown(f"**Ground truth found for `{patient_id}`** — "
                f"IoU (union of boxes) = **{mask_iou(boxes, gt_boxes, cam.shape):.3f}**")
elif annotations_csv and os.path.exists(annotations_csv):
    st.caption(f"No ground-truth boxes for `{patient_id}` (normal case, or not an RSNA training file).")
