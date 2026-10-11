# Pneumonia Detection and Localization

UE24CS352A Machine Learning mini-project. A DenseNet121 classifier predicts pneumonia from chest X-rays in the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
dataset. Grad-CAM heatmaps from the classifier are then converted into approximate bounding boxes, which are
scored against the RSNA ground-truth boxes.

> Educational project. The output is not a clinical diagnosis. The Grad-CAM boxes are approximate
> because the classifier never sees bounding boxes during training.

## Repository layout

```
rc/
├── src/
│   ├── dataset.py        # DICOM loading + PneumoniaDataset                 (Member 1)
│   ├── prepare_data.py   # all_annotations.csv + patient-level 80/10/10 split (Member 1)
│   ├── model.py          # DenseNet121 with a single-logit head              (Member 2)
│   ├── train.py          # BCEWithLogits + Adam, best checkpoint by val AUC  (Member 2)
│   ├── evaluate.py       # held-out test metrics, confusion matrix, ROC      (Member 2)
│   ├── gradcam.py        # Grad-CAM heatmaps from the trained checkpoint     (Member 3)
│   └── localization.py   # heatmap -> boxes, IoU evaluation vs ground truth  (Member 3)
├── app/app.py            # Streamlit demo                                   (Member 3)
├── notebooks/            # Ready-to-run Colab notebooks per member
└── requirements.txt
```

Large files stay out of GitHub. The dataset ZIPs, metadata CSVs, `best_model.pth`, and result plots live
in the shared Google Drive folders `dataset/` and `Pneumonia_Project/` (`metadata/`, `checkpoints/`, `results/`).

## Setup

**Google Colab (recommended, T4 GPU)**

1. In Google Drive, add shortcuts to the shared `dataset` and `Pneumonia_Project` folders under **My Drive**.
2. Open the notebook for your part from `rc/notebooks/` in Colab. Each notebook mounts Drive, clones this
   repository, installs dependencies, and extracts `stage_2_train_images.zip` to `/content/rsna_train_images/`.

**Local**

```bash
git clone https://github.com/Tanujn06/Pneumonia_Detection_Localization.git
cd Pneumonia_Detection_Localization
pip install -r rc/requirements.txt
```

The default script paths point to Colab and Drive locations. When running locally, pass the paths explicitly
with the command-line flags shown below.

## Running the pipeline

All commands run from the repository root.

```bash
# 1. Data: patient-level splits + all_annotations.csv   (Member 1)
python rc/src/prepare_data.py

# 2. Train and evaluate the classifier                 (Member 2)
python rc/src/train.py    --train_csv META/train.csv --val_csv META/val.csv --img_dir IMAGES --checkpoint_dir CKPT_DIR
python rc/src/evaluate.py --test_csv META/test.csv --img_dir IMAGES --checkpoint CKPT_DIR/best_model.pth --results_dir RESULTS

# 3. Grad-CAM for a single image                        (Member 3)
python rc/src/gradcam.py --image IMAGES/<patientId>.dcm --checkpoint CKPT_DIR/best_model.pth --output gradcam.png

# 4. Localization: pick the heatmap threshold on val, then run test once with that threshold
python rc/src/localization.py --split_csv META/val.csv  --annotations_csv META/all_annotations.csv \
    --train_csv META/train.csv --img_dir IMAGES --checkpoint CKPT_DIR/best_model.pth \
    --results_dir RESULTS/localization --thresholds 0.3 0.4 0.5 0.6 0.7
python rc/src/localization.py --split_csv META/test.csv --annotations_csv META/all_annotations.csv \
    --train_csv META/train.csv --img_dir IMAGES --checkpoint CKPT_DIR/best_model.pth \
    --results_dir RESULTS/localization --thresholds <best threshold from val>

# 5. Demo
streamlit run rc/app/app.py
```

To run the Streamlit app in Colab, use Step 9 of `rc/notebooks/Member_3_Localization_and_Demo.ipynb`. It
starts the app and opens a public `trycloudflare.com` link. The app reads its default settings from these
environment variables: `PNEUMONIA_CHECKPOINT`, `PNEUMONIA_ANNOTATIONS`, and `PNEUMONIA_CAM_THRESHOLD`.

## Localization method (Member 3)

1. **Grad-CAM.** The target layer is DenseNet121's last feature map (`features.norm5` followed by ReLU; 7×7
   for a 224×224 input). Each channel is weighted by the spatial mean of the pneumonia logit's gradient. The
   weighted sum goes through a ReLU, is upsampled to the original 1024×1024 image, and is scaled to [0, 1].
   Preprocessing is identical to validation and test: DICOM → `read_xray` → resize to 224 → ImageNet normalization.
2. **Boxes.** The heatmap is thresholded at a fraction of its maximum and split into 8-connected regions.
   Regions smaller than 1% of the image are dropped. The 3 regions with the largest total activation are
   kept, and each one is reported as an `x, y, width, height` box (the RSNA format).
3. **Evaluation** uses every ground-truth-positive patient in the split, whatever the classifier predicted:
   - **Mask IoU:** IoU between the union of the predicted boxes and the union of the ground-truth boxes,
     averaged over images.
   - **Mean best IoU per ground-truth box:** for each ground-truth box, the highest IoU of any predicted box.
   - **Recall and precision at IoU ≥ 0.3 and 0.5:** predicted and ground-truth boxes are matched one to one
     with the Hungarian algorithm.
   - **Pointing game:** whether the heatmap's peak falls inside a ground-truth box.
   - **Prior-box baseline:** one fixed median box per lung, computed from the training-set ground truth.
     The heatmap boxes are only useful if they beat this baseline.
4. **Protocol.** The threshold is chosen on `val.csv` by the best mean mask IoU. `test.csv` is evaluated
   once, with that threshold.

Outputs are written to `Pneumonia_Project/results/localization/`:

| File | Contents |
|---|---|
| `localization_<split>_summary.json` | Aggregate metrics for each threshold, the baseline, and the selected threshold |
| `localization_<split>_per_patient.csv` | Metrics for each image |
| `localization_<split>_threshold_sweep.png` | IoU and recall against the heatmap threshold |
| `localization_<split>_examples.png` | Heatmaps with ground-truth (green) and predicted (red) boxes |

## Team

| Member | Responsibility | Branch |
|---|---|---|
| Member 1 | Dataset and preprocessing | `feature/data` |
| Member 2 | Model training and evaluation | `feature/model` |
| Member 3 | Grad-CAM localization and Streamlit demo | `feature/localization` |
