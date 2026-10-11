"""
localization.py - Approximate pneumonia boxes from Grad-CAM + IoU evaluation
Turns Grad-CAM heatmaps into bounding boxes (threshold -> connected components)
and scores them against RSNA ground-truth boxes from all_annotations.csv.

Grad-CAM is weakly supervised (the classifier never saw a box), so these boxes
are approximate. Choose the heatmap threshold on val.csv, then report test.csv once.
Author: Member 3
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# Ensure imports resolve whether executed from root or rc/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_METADATA_DIR = "/content/drive/MyDrive/Pneumonia_Project/metadata"
DEFAULT_IMG_DIR = "/content/rsna_train_images/stage_2_train_images/"
DEFAULT_RESULTS_DIR = "/content/drive/MyDrive/Pneumonia_Project/results/localization"


# --------------------------------------------------------------------------
# Heatmap -> boxes
# --------------------------------------------------------------------------

def cam_to_boxes(cam, threshold=0.5, min_area_frac=0.01, max_boxes=3):
    """
    Convert a [0, 1] heatmap into boxes in the heatmap's pixel coordinates.

    1. Binarize: cam >= threshold (threshold is relative to the heatmap max of 1).
    2. Label 8-connected regions.
    3. Drop regions smaller than min_area_frac of the image.
    4. Keep the max_boxes regions with the largest total activation.

    Returns a list of dicts {x, y, width, height, score} in RSNA format
    (x, y = top-left corner).
    """
    mask = cam >= threshold
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []

    min_area = min_area_frac * cam.size
    index = np.arange(1, n + 1)
    areas = ndimage.sum(mask, labels, index)
    masses = ndimage.sum(cam, labels, index)
    slices = ndimage.find_objects(labels)

    boxes = []
    for area, mass, slc in zip(areas, masses, slices):
        if area < min_area:
            continue
        ys, xs = slc
        boxes.append({
            "x": float(xs.start),
            "y": float(ys.start),
            "width": float(xs.stop - xs.start),
            "height": float(ys.stop - ys.start),
            "score": float(mass),
        })

    boxes.sort(key=lambda b: b["score"], reverse=True)
    return boxes[:max_boxes]


# --------------------------------------------------------------------------
# IoU and matching
# --------------------------------------------------------------------------

def box_iou(a, b):
    """IoU of two boxes given as dicts with x, y, width, height."""
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(pred_boxes, gt_boxes):
    """Matrix of shape (len(pred), len(gt))."""
    m = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            m[i, j] = box_iou(p, g)
    return m


def match_boxes(pred_boxes, gt_boxes):
    """
    One-to-one matching (Hungarian algorithm, maximizing total IoU).
    Returns the IoU of each matched (pred, gt) pair.
    """
    if not pred_boxes or not gt_boxes:
        return []
    m = iou_matrix(pred_boxes, gt_boxes)
    rows, cols = linear_sum_assignment(-m)
    return [float(m[r, c]) for r, c in zip(rows, cols)]


def boxes_to_mask(boxes, shape):
    mask = np.zeros(shape, dtype=bool)
    for b in boxes:
        x1, y1 = int(round(b["x"])), int(round(b["y"]))
        x2, y2 = int(round(b["x"] + b["width"])), int(round(b["y"] + b["height"]))
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
    return mask


def mask_iou(pred_boxes, gt_boxes, shape):
    """IoU between the union of predicted boxes and the union of GT boxes (pixel level)."""
    p = boxes_to_mask(pred_boxes, shape)
    g = boxes_to_mask(gt_boxes, shape)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union > 0 else 0.0


def pointing_hit(cam, gt_boxes):
    """Pointing game: does the heatmap's peak fall inside any GT box?"""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return any(
        b["x"] <= x < b["x"] + b["width"] and b["y"] <= y < b["y"] + b["height"]
        for b in gt_boxes
    )


def score_image(pred_boxes, gt_boxes, shape):
    """Per-image localization metrics for one set of predicted boxes."""
    matched = match_boxes(pred_boxes, gt_boxes)
    best_per_gt = iou_matrix(pred_boxes, gt_boxes).max(axis=0) if pred_boxes else np.zeros(len(gt_boxes))
    return {
        "n_pred": len(pred_boxes),
        "n_gt": len(gt_boxes),
        "mask_iou": mask_iou(pred_boxes, gt_boxes, shape),
        "sum_best_iou": float(best_per_gt.sum()),
        "tp_03": int(sum(i >= 0.3 for i in matched)),
        "tp_05": int(sum(i >= 0.5 for i in matched)),
    }


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def load_gt_boxes(annotations_csv):
    """patientId -> list of GT boxes (only Target == 1 rows carry boxes)."""
    ann = pd.read_csv(annotations_csv)
    ann = ann[ann["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    gt = {}
    for row in ann.itertuples(index=False):
        gt.setdefault(row.patientId, []).append({
            "x": float(row.x), "y": float(row.y),
            "width": float(row.width), "height": float(row.height),
        })
    return gt


def prior_baseline_boxes(train_ids, gt):
    """
    Non-learned baseline: one median box per lung, computed from TRAIN GT boxes only.
    Any useful heatmap localization should beat this.
    """
    left, right = [], []
    for pid in train_ids:
        for b in gt.get(pid, []):
            cx = b["x"] + b["width"] / 2
            (left if cx < 512 else right).append([b["x"], b["y"], b["width"], b["height"]])
    boxes = []
    for group in (left, right):
        if group:
            x, y, w, h = np.median(np.array(group), axis=0)
            boxes.append({"x": x, "y": y, "width": w, "height": h, "score": 1.0})
    return boxes


def summarize(rows):
    """Aggregate per-image rows (one threshold) into dataset-level metrics."""
    df = pd.DataFrame(rows)
    total_gt, total_pred = df["n_gt"].sum(), df["n_pred"].sum()
    return {
        "n_images": int(len(df)),
        "n_gt_boxes": int(total_gt),
        "n_pred_boxes": int(total_pred),
        "mean_mask_iou": float(df["mask_iou"].mean()),
        "mean_best_iou_per_gt_box": float(df["sum_best_iou"].sum() / total_gt) if total_gt else 0.0,
        "recall@0.3": float(df["tp_03"].sum() / total_gt) if total_gt else 0.0,
        "recall@0.5": float(df["tp_05"].sum() / total_gt) if total_gt else 0.0,
        "precision@0.3": float(df["tp_03"].sum() / total_pred) if total_pred else 0.0,
        "precision@0.5": float(df["tp_05"].sum() / total_pred) if total_pred else 0.0,
        "images_with_no_box": int((df["n_pred"] == 0).sum()),
    }


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def draw_boxes(ax, boxes, color, label=None):
    import matplotlib.patches as patches
    for i, b in enumerate(boxes):
        ax.add_patch(patches.Rectangle(
            (b["x"], b["y"]), b["width"], b["height"],
            fill=False, edgecolor=color, linewidth=2,
            label=label if i == 0 else None,
        ))


def save_example_figure(examples, threshold, args, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gradcam import overlay_heatmap

    n = len(examples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, ex in zip(axes.flat, examples):
        cam = ex["cam"]
        boxes = cam_to_boxes(cam, threshold, args.min_area_frac, args.max_boxes)
        ax.imshow(overlay_heatmap(ex["image"], cam))
        draw_boxes(ax, ex["gt"], "lime", "Ground truth")
        draw_boxes(ax, boxes, "red", "Grad-CAM box")
        iou = mask_iou(boxes, ex["gt"], cam.shape)
        ax.set_title(f"{ex['pid'][:8]}  p={ex['prob']:.2f}  IoU={iou:.2f}", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"Grad-CAM localization (heatmap threshold = {threshold})")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def save_threshold_plot(summaries, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ths = sorted(summaries)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ["mean_mask_iou", "mean_best_iou_per_gt_box", "recall@0.3", "recall@0.5"]:
        ax.plot(ths, [summaries[t][key] for t in ths], marker="o", label=key)
    ax.set_xlabel("Grad-CAM threshold (fraction of max activation)")
    ax.set_ylabel("Score")
    ax.set_title("Localization quality vs heatmap threshold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Evaluation CLI
# --------------------------------------------------------------------------

def evaluate():
    import torch
    from PIL import Image
    from gradcam import load_model, load_image, explain, DEFAULT_CHECKPOINT

    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM boxes against RSNA ground truth (IoU)")
    parser.add_argument("--split_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/val.csv",
                        help="Split to evaluate (use val.csv to pick the threshold, then test.csv once)")
    parser.add_argument("--annotations_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/all_annotations.csv")
    parser.add_argument("--train_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/train.csv",
                        help="Used only to build the prior-box baseline (set to '' to skip)")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7],
                        help="Heatmap thresholds to sweep")
    parser.add_argument("--min_area_frac", type=float, default=0.01)
    parser.add_argument("--max_boxes", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=0, help="Limit #images (0 = all positives)")
    parser.add_argument("--num_examples", type=int, default=8, help="Example images to plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member 3 Localization] Device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)
    split_name = os.path.splitext(os.path.basename(args.split_csv))[0]

    model = load_model(args.checkpoint, device)
    gt = load_gt_boxes(args.annotations_csv)

    split = pd.read_csv(args.split_csv)
    positives = [pid for pid in split.loc[split["Target"] == 1, "patientId"] if pid in gt]
    if args.max_images:
        positives = positives[:args.max_images]
    print(f"Evaluating {len(positives)} ground-truth-positive patients from {args.split_csv}")

    baseline = []
    if args.train_csv and os.path.exists(args.train_csv):
        baseline = prior_baseline_boxes(pd.read_csv(args.train_csv)["patientId"], gt)

    rows = {t: [] for t in args.thresholds}
    baseline_rows, per_patient, examples = [], [], []
    pointing_hits = 0
    probs = []

    for i, pid in enumerate(positives, 1):
        image = load_image(os.path.join(args.img_dir, f"{pid}.dcm"))
        cam, prob = explain(model, image, device)
        gt_boxes = gt[pid]
        hit = pointing_hit(cam, gt_boxes)
        pointing_hits += hit
        probs.append(prob)

        for t in args.thresholds:
            boxes = cam_to_boxes(cam, t, args.min_area_frac, args.max_boxes)
            s = score_image(boxes, gt_boxes, cam.shape)
            rows[t].append(s)
            per_patient.append({"patientId": pid, "prob": prob, "threshold": t,
                                "pointing_hit": int(hit), **s})
        if baseline:
            baseline_rows.append(score_image(baseline, gt_boxes, cam.shape))

        if len(examples) < args.num_examples:
            small = (256, 256)
            scale = 256 / cam.shape[1], 256 / cam.shape[0]
            examples.append({
                "pid": pid, "prob": prob,
                "image": image.resize(small, Image.BILINEAR),
                "cam": np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(small), dtype=np.float32) / 255,
                "gt": [{"x": b["x"] * scale[0], "y": b["y"] * scale[1],
                        "width": b["width"] * scale[0], "height": b["height"] * scale[1]} for b in gt_boxes],
            })

        if i % 50 == 0 or i == len(positives):
            print(f"  processed {i}/{len(positives)}")

    if not positives:
        raise SystemExit("No positive patients with boxes found in this split.")

    summaries = {t: summarize(rows[t]) for t in args.thresholds}
    best_t = max(summaries, key=lambda t: summaries[t]["mean_mask_iou"])
    probs = np.array(probs)

    report = {
        "split": split_name,
        "checkpoint": args.checkpoint,
        "method": {
            "heatmap": "Grad-CAM on DenseNet121 features (norm5 + ReLU), upsampled to original resolution",
            "boxes": "threshold relative to heatmap max -> 8-connected components -> "
                     f"drop regions < {args.min_area_frac:.0%} of image -> keep top {args.max_boxes} by total activation",
            "matching": "one-to-one Hungarian matching on box IoU; recall/precision count matched pairs with IoU >= 0.3 / 0.5",
            "mask_iou": "IoU between union of predicted boxes and union of GT boxes, averaged per image",
            "population": "all ground-truth-positive patients in the split (regardless of classifier output)",
        },
        "classifier_on_these_positives": {
            "mean_prob": float(probs.mean()),
            "sensitivity@0.5": float((probs >= 0.5).mean()),
        },
        "pointing_game_accuracy": pointing_hits / len(positives),
        "best_threshold_by_mean_mask_iou": best_t,
        "per_threshold": {str(t): s for t, s in summaries.items()},
    }
    if baseline_rows:
        report["prior_box_baseline"] = summarize(baseline_rows)

    summary_path = os.path.join(args.results_dir, f"localization_{split_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=4)
    per_patient_path = os.path.join(args.results_dir, f"localization_{split_name}_per_patient.csv")
    pd.DataFrame(per_patient).to_csv(per_patient_path, index=False)
    sweep_path = os.path.join(args.results_dir, f"localization_{split_name}_threshold_sweep.png")
    save_threshold_plot(summaries, sweep_path)
    examples_path = os.path.join(args.results_dir, f"localization_{split_name}_examples.png")
    save_example_figure(examples, best_t, args, examples_path)

    print("\n" + "=" * 72)
    print(f"  GRAD-CAM LOCALIZATION ({split_name}, {len(positives)} positive patients)")
    print("=" * 72)
    print(f"  Pointing game accuracy: {report['pointing_game_accuracy']:.4f}")
    print(f"  {'thr':>5} {'maskIoU':>8} {'bestIoU':>8} {'R@0.3':>7} {'R@0.5':>7} {'P@0.3':>7} {'P@0.5':>7} {'noBox':>6}")
    for t in args.thresholds:
        s = summaries[t]
        print(f"  {t:>5.2f} {s['mean_mask_iou']:>8.4f} {s['mean_best_iou_per_gt_box']:>8.4f} "
              f"{s['recall@0.3']:>7.4f} {s['recall@0.5']:>7.4f} {s['precision@0.3']:>7.4f} "
              f"{s['precision@0.5']:>7.4f} {s['images_with_no_box']:>6}")
    if baseline_rows:
        b = report["prior_box_baseline"]
        print(f"  {'prior':>5} {b['mean_mask_iou']:>8.4f} {b['mean_best_iou_per_gt_box']:>8.4f} "
              f"{b['recall@0.3']:>7.4f} {b['recall@0.5']:>7.4f} {b['precision@0.3']:>7.4f} "
              f"{b['precision@0.5']:>7.4f} {b['images_with_no_box']:>6}")
    print("-" * 72)
    print(f"  Best threshold on this split (mean mask IoU): {best_t}")
    print("=" * 72)
    print(f"Artifacts saved to {args.results_dir}:")
    for p in [summary_path, per_patient_path, sweep_path, examples_path]:
        print(f"  -> {p}")


if __name__ == "__main__":
    evaluate()
