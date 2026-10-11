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

# 4. Localization: compare methods/thresholds on val, then run test once with the val-selected setting
python rc/src/localization.py --split_csv META/val.csv  --annotations_csv META/all_annotations.csv \
    --train_csv META/train.csv --img_dir IMAGES --checkpoint CKPT_DIR/best_model.pth \
    --results_dir RESULTS/localization
python rc/src/localization.py --split_csv META/test.csv --annotations_csv META/all_annotations.csv \
    --train_csv META/train.csv --img_dir IMAGES --checkpoint CKPT_DIR/best_model.pth \
    --results_dir RESULTS/localization --use_val_selection RESULTS/localization/localization_val_summary.json

# 5. Demo
streamlit run rc/app/app.py
```

To run the Streamlit app in Colab, use Step 9 of `rc/notebooks/Member_3_Localization_and_Demo.ipynb`. It
starts the app and opens a public `trycloudflare.com` link. The app reads its default settings from these
environment variables: `PNEUMONIA_CHECKPOINT`, `PNEUMONIA_ANNOTATIONS`, `PNEUMONIA_CAM_THRESHOLD`, and
`PNEUMONIA_LOC_CONFIG` (for example `denseblock3+tta+prior+split`).

## Localization method (Member 3)

1. **Grad-CAM.** Each channel of a DenseNet121 feature map is weighted by the spatial mean of the pneumonia
   logit's gradient. The weighted sum goes through a ReLU, is upsampled to the original 1024×1024 image, and is
   scaled to [0, 1]. Preprocessing is identical to validation and test: DICOM → `read_xray` → resize to 224 →
   ImageNet normalization. Two target layers are supported: `norm5` (the last feature map, 7×7, the original
   method) and `denseblock3` (14×14, finer).
2. **Boxes.** The heatmap is thresholded at a fraction of its maximum and split into 8-connected regions.
   Regions smaller than 1% of the image are dropped. The 3 regions with the largest total activation are
   kept, and each one is reported as an `x, y, width, height` box (the RSNA format).
3. **Optional improvements**, combined into a configuration string such as `denseblock3+tta+prior+split`:
   - `tta`: average the heatmap with the heatmap of the horizontally mirrored image.
   - `prior`: multiply the heatmap by a map of where pneumonia boxes occur in the **train** split. This
     down-weights regions such as the shoulders or the image border.
   - `split`: cut the mask at the vertical midline so each lung gets its own box. 54% of positive RSNA
     patients have two boxes, one per lung.
4. **Evaluation** uses every ground-truth-positive patient in the split, whatever the classifier predicted:
   - **Mask IoU:** IoU between the union of the predicted boxes and the union of the ground-truth boxes,
     averaged over images.
   - **Mean best IoU per ground-truth box:** for each ground-truth box, the highest IoU of any predicted box.
   - **Recall and precision at IoU ≥ 0.3 and 0.5:** predicted and ground-truth boxes are matched one to one
     with the Hungarian algorithm.
   - **Pointing game:** whether the heatmap's peak falls inside a ground-truth box.
   - **References that use no model:** one fixed median box per lung, and the train location map on its own.
     The heatmap boxes are only useful if they beat these.
   - **Coordinate check:** image sizes and the number of ground-truth boxes that fall outside the image.
5. **Protocol.** Every configuration and threshold is compared on `val.csv`, and the setting with the best mean
   mask IoU is selected. `test.csv` is evaluated once with that setting, alongside the original `norm5` method.

Outputs are written to `Pneumonia_Project/results/localization/`:

| File | Contents |
|---|---|
| `localization_<split>_summary.json` | Selected setting, original method, references that use no model, coordinate check |
| `localization_<split>_comparison.csv` | Metrics for every configuration and threshold |
| `localization_<split>_per_patient.csv` | Metrics for each image |
| `localization_<split>_threshold_sweep.png` | Mean mask IoU against threshold for each configuration |
| `localization_<split>_examples.png` | Original and selected method on the same patients, with ground-truth (green) and predicted (red) boxes |

## Team

| Member | Responsibility | Branch |
|---|---|---|
| Member 1 | Dataset and preprocessing | `feature/data` |
| Member 2 | Model training and evaluation | `feature/model` |
| Member 3 | Grad-CAM localization and Streamlit demo | `feature/localization` |
