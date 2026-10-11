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

def cam_to_boxes(cam, threshold=0.5, min_area_frac=0.01, max_boxes=3, split_midline=False):
    """
    Convert a [0, 1] heatmap into boxes in the heatmap's pixel coordinates.

    1. Binarize: cam >= threshold (threshold is relative to the heatmap max of 1).
    2. Optionally cut the mask at the vertical midline, so a blob spanning both
       lungs becomes one box per lung (RSNA annotates each lung separately).
    3. Label 8-connected regions.
    4. Drop regions smaller than min_area_frac of the image.
    5. Keep the max_boxes regions with the largest total activation.

    Returns a list of dicts {x, y, width, height, score} in RSNA format
    (x, y = top-left corner).
    """
    mask = cam >= threshold
    if split_midline:
        mid = cam.shape[1] // 2
        mask[:, mid - 1:mid + 1] = False
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


def scale_boxes(boxes, sx, sy):
    """Rescale boxes from one pixel grid to another (sx, sy = new / old size)."""
    return [{**b, "x": b["x"] * sx, "y": b["y"] * sy,
             "width": b["width"] * sx, "height": b["height"] * sy} for b in boxes]


def prior_baseline_boxes(train_ids, gt, image_size=1024):
    """
    Non-learned baseline: one median box per lung, computed from TRAIN GT boxes only.
    Any useful heatmap localization should beat this.
    """
    left, right = [], []
    for pid in train_ids:
        for b in gt.get(pid, []):
            cx = b["x"] + b["width"] / 2
            (left if cx < image_size / 2 else right).append([b["x"], b["y"], b["width"], b["height"]])
    boxes = []
    for group in (left, right):
        if group:
            x, y, w, h = np.median(np.array(group), axis=0)
            boxes.append({"x": x, "y": y, "width": w, "height": h, "score": 1.0})
    return boxes


def location_prior(train_ids, gt, size=256, image_size=1024):
    """
    Where pneumonia boxes occur: per-pixel count of TRAIN GT boxes covering it,
    scaled to [0, 1] on a size x size grid. Built from the train split only, so
    using it on val/test does not leak labels.
    """
    density = np.zeros((size, size), dtype=np.float32)
    s = size / image_size
    for pid in train_ids:
        for b in gt.get(pid, []):
            x1, y1 = int(b["x"] * s), int(b["y"] * s)
            x2, y2 = int(np.ceil((b["x"] + b["width"]) * s)), int(np.ceil((b["y"] + b["height"]) * s))
            density[y1:y2, x1:x2] += 1
    return density / density.max() if density.max() > 0 else density


def apply_prior(cam, prior):
    """Down-weight heatmap regions where pneumonia is rarely annotated, then renormalize."""
    if prior.shape != cam.shape:
        from PIL import Image
        prior = np.asarray(Image.fromarray(prior).resize(cam.shape[::-1], Image.BILINEAR))
    weighted = cam * prior
    return weighted / weighted.max() if weighted.max() > 0 else weighted


# --------------------------------------------------------------------------
# Localization configurations
# --------------------------------------------------------------------------

# A config is "<layer>[+tta][+prior][+split]", e.g. "denseblock3+tta+prior+split".
#   layer : Grad-CAM target layer (norm5 = 7x7, denseblock3 = 14x14)
#   tta   : average with the horizontally flipped image's heatmap
#   prior : multiply by the train-set location prior
#   split : cut boxes at the vertical midline (one box per lung)
BASELINE_CONFIG = "norm5"
DEFAULT_CONFIGS = [
    "norm5", "norm5+split", "norm5+tta+split", "norm5+prior+split", "norm5+tta+prior+split",
    "denseblock3", "denseblock3+split", "denseblock3+tta+split", "denseblock3+prior+split",
    "denseblock3+tta+prior+split",
]
OPTIONS = {"tta", "prior", "split"}


def parse_config(config):
    layer, *opts = config.split("+")
    unknown = set(opts) - OPTIONS
    if unknown:
        raise ValueError(f"Unknown option(s) {unknown} in config '{config}'")
    return {"layer": layer, "tta": "tta" in opts, "prior": "prior" in opts, "split": "split" in opts}


def localize(cams, config, threshold, prior=None, min_area_frac=0.01, max_boxes=3):
    """
    Apply one config to precomputed heatmaps.
    cams: {(layer, tta): cam} as returned by gradcam.explain_multi.
    Returns (heatmap used, boxes) in the heatmap's pixel grid.
    """
    c = parse_config(config)
    cam = cams[(c["layer"], c["tta"])]
    if c["prior"]:
        if prior is None:
            raise ValueError(f"Config '{config}' needs a location prior (train.csv + annotations)")
        cam = apply_prior(cam, prior)
    return cam, cam_to_boxes(cam, threshold, min_area_frac, max_boxes, c["split"])


def summarize(rows):
    """Aggregate per-image rows (one config + threshold) into dataset-level metrics."""
    df = pd.DataFrame(rows)
    total_gt, total_pred = df["n_gt"].sum(), df["n_pred"].sum()
    out = {
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
    if "pointing_hit" in df:
        out["pointing_game"] = float(df["pointing_hit"].mean())
    return out


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


def save_example_figure(examples, settings, prior, args, output_path):
    """One row per (config, threshold) in settings, one column per example patient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gradcam import overlay_heatmap

    rows, cols = len(settings), len(examples)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for r, (config, threshold) in enumerate(settings):
        for c, ex in enumerate(examples):
            ax = axes[r, c]
            cam, boxes = localize(ex["cams"], config, threshold, prior, args.min_area_frac, args.max_boxes)
            ax.imshow(overlay_heatmap(ex["image"], cam))
            draw_boxes(ax, ex["gt"], "lime", "Radiologist box (ground truth)")
            draw_boxes(ax, boxes, "red", "Grad-CAM box")
            iou = mask_iou(boxes, ex["gt"], cam.shape)
            ax.set_title(f"{ex['pid'][:8]} p={ex['prob']:.2f} IoU={iou:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{config}\nthr={threshold}", fontsize=9)

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Grad-CAM localization vs radiologist annotations")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def save_threshold_plot(table, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for config, group in table.groupby("config", sort=False):
        group = group.sort_values("threshold")
        ax.plot(group["threshold"], group["mean_mask_iou"], marker="o", label=config,
                linewidth=2.5 if config == BASELINE_CONFIG else 1.5)
    ax.set_xlabel("Heatmap threshold (fraction of max activation)")
    ax.set_ylabel("Mean mask IoU")
    ax.set_title("Localization quality per configuration and threshold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Evaluation CLI
# --------------------------------------------------------------------------

def evaluate():
    import torch
    from collections import Counter
    from PIL import Image
    from gradcam import load_model, load_image, explain_multi, DEFAULT_CHECKPOINT

    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM boxes against RSNA ground truth (IoU)")
    parser.add_argument("--split_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/val.csv",
                        help="Split to evaluate (use val.csv to choose settings, then test.csv once)")
    parser.add_argument("--annotations_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/all_annotations.csv")
    parser.add_argument("--train_csv", type=str, default=f"{DEFAULT_METADATA_DIR}/train.csv",
                        help="Train split, used only for the location prior and the baselines")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS,
                        help="Configs '<layer>[+tta][+prior][+split]' to compare")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help="Heatmap thresholds to sweep")
    parser.add_argument("--use_val_selection", type=str, default="",
                        help="Path to localization_val_summary.json: evaluate only the val-selected "
                             "config/threshold and the baseline at its own val-selected threshold")
    parser.add_argument("--min_area_frac", type=float, default=0.01)
    parser.add_argument("--max_boxes", type=int, default=3)
    parser.add_argument("--eval_size", type=int, default=256,
                        help="Heatmaps and GT boxes are rescaled to this grid for scoring (IoU is scale-invariant)")
    parser.add_argument("--max_images", type=int, default=0, help="Limit #images (0 = all positives)")
    parser.add_argument("--num_examples", type=int, default=6, help="Example patients to plot")
    args = parser.parse_args()

    # Which (config, threshold) pairs to score
    if args.use_val_selection:
        with open(args.use_val_selection) as f:
            val_report = json.load(f)
        best = val_report["best"]
        base = val_report.get("baseline") or best
        settings = [(best["config"], best["threshold"])]
        if (base["config"], base["threshold"]) != settings[0]:
            settings.append((base["config"], base["threshold"]))
    else:
        settings = [(c, t) for c in args.configs for t in args.thresholds]
    configs = list(dict.fromkeys(c for c, _ in settings))
    parsed = {c: parse_config(c) for c in configs}
    layers = sorted({p["layer"] for p in parsed.values()})
    need_tta = any(p["tta"] for p in parsed.values())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member 3 Localization] Device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)
    split_name = os.path.splitext(os.path.basename(args.split_csv))[0]

    model = load_model(args.checkpoint, device)
    gt = load_gt_boxes(args.annotations_csv)
    train_ids = pd.read_csv(args.train_csv)["patientId"] if args.train_csv and os.path.exists(args.train_csv) else []
    if any(p["prior"] for p in parsed.values()) and len(train_ids) == 0:
        raise SystemExit("Configs with '+prior' need --train_csv")
    size = args.eval_size
    prior = location_prior(train_ids, gt, size) if len(train_ids) else None
    baseline_boxes = prior_baseline_boxes(train_ids, gt) if len(train_ids) else []

    split = pd.read_csv(args.split_csv)
    positives = [pid for pid in split.loc[split["Target"] == 1, "patientId"] if pid in gt]
    if args.max_images:
        positives = positives[:args.max_images]
    if not positives:
        raise SystemExit("No positive patients with boxes found in this split.")
    print(f"Evaluating {len(positives)} ground-truth-positive patients from {args.split_csv}")
    print(f"Comparing {len(configs)} config(s) x thresholds = {len(settings)} settings")

    rows = {s: [] for s in settings}
    reference_rows = {"median box per lung (train)": [], "location prior only (train)": []}
    per_patient, examples, probs = [], [], []
    image_sizes, out_of_bounds = Counter(), 0

    for i, pid in enumerate(positives, 1):
        image = load_image(os.path.join(args.img_dir, f"{pid}.dcm"))
        w, h = image.size
        image_sizes[f"{w}x{h}"] += 1
        # Coordinate check: GT boxes are in original DICOM pixels, same grid as the image.
        out_of_bounds += sum(b["x"] < 0 or b["y"] < 0 or b["x"] + b["width"] > w or b["y"] + b["height"] > h
                             for b in gt[pid])

        cams, prob = explain_multi(model, image, device, layers, need_tta)
        cams = {k: np.asarray(Image.fromarray(v).resize((size, size), Image.BILINEAR)) for k, v in cams.items()}
        gt_boxes = scale_boxes(gt[pid], size / w, size / h)
        probs.append(prob)

        for config, threshold in settings:
            cam, boxes = localize(cams, config, threshold, prior, args.min_area_frac, args.max_boxes)
            s = score_image(boxes, gt_boxes, cam.shape)
            s["pointing_hit"] = int(pointing_hit(cam, gt_boxes))
            s["detected"] = int(prob >= 0.5)
            rows[(config, threshold)].append(s)
            per_patient.append({"patientId": pid, "prob": prob, "config": config, "threshold": threshold, **s})

        if baseline_boxes:
            reference_rows["median box per lung (train)"].append(
                score_image(scale_boxes(baseline_boxes, size / w, size / h), gt_boxes, (size, size)))
        if prior is not None:
            reference_rows["location prior only (train)"].append(
                score_image(cam_to_boxes(prior, 0.5, args.min_area_frac, args.max_boxes, True), gt_boxes, (size, size)))

        if len(examples) < args.num_examples:
            examples.append({"pid": pid, "prob": prob, "cams": cams, "gt": gt_boxes,
                             "image": image.resize((size, size), Image.BILINEAR)})
        if i % 50 == 0 or i == len(positives):
            print(f"  processed {i}/{len(positives)}")

    # Comparison table: one row per (config, threshold)
    table = pd.DataFrame([{"config": c, "threshold": t, **summarize(rows[(c, t)])} for c, t in settings])
    if args.use_val_selection:
        # Settings were fixed on val; never re-select on the split being reported.
        best = {"config": settings[0][0], "threshold": settings[0][1]}
        baseline = {"config": settings[-1][0], "threshold": settings[-1][1]}
    else:
        best_idx = table["mean_mask_iou"].idxmax()
        best = {"config": table.loc[best_idx, "config"], "threshold": float(table.loc[best_idx, "threshold"])}
        base_table = table[table["config"] == BASELINE_CONFIG]
        baseline = ({"config": BASELINE_CONFIG,
                     "threshold": float(base_table.loc[base_table["mean_mask_iou"].idxmax(), "threshold"])}
                    if len(base_table) else None)
    references = {name: summarize(r) for name, r in reference_rows.items() if r}
    probs = np.array(probs)

    def with_detected(setting):
        """Metrics on all positives, plus on those the classifier detects (p >= 0.5)."""
        out = summarize(rows[setting])
        detected = [r for r in rows[setting] if r["detected"]]
        out["on_classifier_detected"] = summarize(detected) if detected else None
        return out

    report = {
        "split": split_name,
        "checkpoint": args.checkpoint,
        "n_patients": len(positives),
        "coordinate_check": {
            "image_sizes": dict(image_sizes),
            "gt_boxes_outside_image": int(out_of_bounds),
            "note": "GT boxes (all_annotations.csv) and heatmaps are both in original-image pixels; "
                    f"both are rescaled to {size}x{size} for scoring",
        },
        "method": {
            "heatmap": "Grad-CAM of the pneumonia logit; layer norm5 (7x7) or denseblock3 (14x14)",
            "options": "tta = average with mirrored image; prior = multiply by train-set box density; "
                       "split = cut at vertical midline (one box per lung)",
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
        "best": {**best, **with_detected((best["config"], best["threshold"]))},
        "baseline": ({**baseline, **with_detected((baseline["config"], baseline["threshold"]))}
                     if baseline else None),
        "image_free_references": references,
    }

    summary_path = os.path.join(args.results_dir, f"localization_{split_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=4)
    table_path = os.path.join(args.results_dir, f"localization_{split_name}_comparison.csv")
    table.to_csv(table_path, index=False)
    per_patient_path = os.path.join(args.results_dir, f"localization_{split_name}_per_patient.csv")
    pd.DataFrame(per_patient).to_csv(per_patient_path, index=False)
    sweep_path = os.path.join(args.results_dir, f"localization_{split_name}_threshold_sweep.png")
    save_threshold_plot(table, sweep_path)
    examples_path = os.path.join(args.results_dir, f"localization_{split_name}_examples.png")
    shown = [(baseline["config"], baseline["threshold"])] if baseline else []
    if (best["config"], best["threshold"]) not in shown:
        shown.append((best["config"], best["threshold"]))
    save_example_figure(examples, shown, prior, args, examples_path)

    cols = ["mean_mask_iou", "mean_best_iou_per_gt_box", "recall@0.3", "recall@0.5",
            "precision@0.3", "precision@0.5", "pointing_game"]
    heads = ["maskIoU", "bestIoU", "R@0.3", "R@0.5", "P@0.3", "P@0.5", "point"]

    def line(name, s):
        return f"  {name:34s}" + "".join(f" {s.get(c, float('nan')):>7.3f}" for c in cols)

    print("\n" + "=" * 92)
    print(f"  GRAD-CAM LOCALIZATION ({split_name}, {len(positives)} positive patients)")
    print("=" * 92)
    print(f"  Coordinate check: image sizes {dict(image_sizes)}, GT boxes outside image: {out_of_bounds}")
    print(f"  {'setting (best threshold per config)':34s}" + "".join(f" {h:>7s}" for h in heads))
    for config in configs:
        sub = table[table["config"] == config]
        r = sub.loc[sub["mean_mask_iou"].idxmax()]
        print(line(f"{config} @ {r['threshold']:.2f}", r))
    for name, s in references.items():
        print(line(f"[no model] {name}", s))
    print("-" * 92)
    chosen = "selected on val" if args.use_val_selection else f"best on {split_name}"
    print(f"  {chosen}: {best['config']} @ threshold {best['threshold']}")
    if baseline:
        print(f"  Original method:  {baseline['config']} @ threshold {baseline['threshold']}")
    detected = report["best"]["on_classifier_detected"]
    if detected:
        print(line(f"selected, only p>=0.5 ({detected['n_images']} pts)", detected))
    print("=" * 92)
    print(f"Artifacts saved to {args.results_dir}:")
    for p in [summary_path, table_path, per_patient_path, sweep_path, examples_path]:
        print(f"  -> {p}")


if __name__ == "__main__":
    evaluate()
