
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# Project paths and imports
# --------------------------------------------------
RC_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RC_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gradcam import load_model, load_image, explain, overlay_heatmap

DEFAULT_CHECKPOINT = os.environ.get(
    "PNEUMONIA_CHECKPOINT",
    "/content/drive/MyDrive/Pneumonia_Project/checkpoint1/best_model.pth",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Streamlit page
# --------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection & Localization",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Pneumonia Detection & Localization")
st.write(
    "Upload a chest X-ray to obtain a pneumonia classification "
    "and a Grad-CAM-based visual explanation."
)

st.info(
    "This is an academic demonstration. Predictions and highlighted "
    "regions are not a substitute for professional medical diagnosis."
)

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
st.sidebar.header("Model settings")

checkpoint_path = st.sidebar.text_input(
    "Trained checkpoint path",
    value=DEFAULT_CHECKPOINT,
)

target_layer = st.sidebar.selectbox(
    "Grad-CAM layer",
    options=["norm5", "denseblock3"],
    index=0,
)

flip_tta = st.sidebar.checkbox(
    "Use horizontal-flip TTA for heatmap",
    value=True,
)

heatmap_threshold = st.sidebar.slider(
    "Localization threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.05,
)

st.sidebar.caption(f"Device: {DEVICE}")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def get_model(path, device_name):
    return load_model(path, torch.device(device_name))


# --------------------------------------------------
# Convert heatmap into approximate bounding boxes
# --------------------------------------------------
def heatmap_to_boxes(cam, threshold=0.6, min_area_frac=0.01, max_boxes=3):
    """
    Generate approximate boxes from a normalized Grad-CAM heatmap.
    These are not boxes directly predicted by a detector.
    """
    cam = np.asarray(cam, dtype=np.float32)

    if cam.size == 0 or float(cam.max()) <= 0:
        return []

    mask = (cam >= threshold * cam.max()).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    min_area = min_area_frac * cam.shape[0] * cam.shape[1]
    boxes = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        boxes.append({
            "box": (int(x), int(y), int(x + w), int(y + h)),
            "area": int(area),
            "activation": float(cam[labels == i].mean()),
        })

    boxes.sort(key=lambda item: item["activation"], reverse=True)
    return boxes[:max_boxes]


def draw_boxes(image, boxes):
    result = np.asarray(image.convert("RGB")).copy()

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return result


# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="Supported formats: PNG, JPG, JPEG and DICOM (.dcm).",
)

if uploaded is not None:
    st.subheader("Input image")

    # gradcam.load_image supports DICOM and ordinary image files.
    suffix = Path(uploaded.name).suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(uploaded.getvalue())
        temp_path = temp.name

    try:
        image = load_image(temp_path)

        left, right = st.columns([1, 1])

        with left:
            st.image(image, caption="Uploaded chest X-ray", width="stretch")

        if st.button("Analyze X-ray", type="primary", use_container_width=True):
            if not Path(checkpoint_path).is_file():
                st.error(
                    "Checkpoint not found. Enter the correct path in the "
                    "sidebar and try again."
                )
            else:
                try:
                    with st.spinner("Loading model and generating explanation..."):
                        model = get_model(checkpoint_path, str(DEVICE))

                        cam, probability = explain(
                            model,
                            image,
                            DEVICE,
                            target_layer=target_layer,
                            flip_tta=flip_tta,
                        )

                        heatmap_overlay = overlay_heatmap(
                            image, cam, alpha=0.4
                        )

                        boxes = heatmap_to_boxes(
                            cam, threshold=heatmap_threshold
                        )

                        localized_image = draw_boxes(image, boxes)

                    st.subheader("Classification result")

                    metric1, metric2 = st.columns(2)

                    with metric1:
                        st.metric(
                            "Pneumonia probability",
                            f"{probability * 100:.2f}%",
                        )

                    with metric2:
                        st.metric(
                            "No-pneumonia probability",
                            f"{(1 - probability) * 100:.2f}%",
                        )

                    if probability >= 0.5:
                        st.warning(
                            "Model prediction: Pneumonia-positive"
                        )
                    else:
                        st.success(
                            "Model prediction: Pneumonia-negative"
                        )

                    st.subheader("Grad-CAM visualization")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.image(
                            heatmap_overlay,
                            caption="Grad-CAM heatmap overlay",
                            width="stretch",
                        )

                    with c2:
                        st.image(
                            localized_image,
                            caption="Approximate localization boxes",
                            width="stretch",
                        )

                    st.caption(
                        "Green boxes are generated from thresholded Grad-CAM "
                        "activation regions. They are approximate visual "
                        "localizations, not ground-truth boxes or clinical "
                        "segmentation."
                    )

                    if boxes:
                        st.write("Approximate regions:")
                        for index, item in enumerate(boxes, start=1):
                            st.write(
                                f"Region {index}: {item['box']}"
                            )
                    else:
                        st.info(
                            "No heatmap region met the current threshold. "
                            "Try adjusting the localization threshold."
                        )

                    with st.expander("Technical details"):
                        st.write(f"Device: {DEVICE}")
                        st.write(f"Grad-CAM layer: {target_layer}")
                        st.write(f"Flip TTA: {flip_tta}")
                        st.write(f"Heatmap threshold: {heatmap_threshold}")

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.exception(exc)

    except Exception as exc:
        st.error(f"Could not read the uploaded X-ray: {exc}")

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

st.divider()
st.caption(
    "Deep Learning Pneumonia Detection & Localization | "
    "DenseNet121 + Grad-CAM | Academic prototype"
)
