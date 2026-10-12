# Pneumonia Detection with Weakly Supervised Localization

UE24CS352A Machine Learning - Mini-Project

A CNN (DenseNet121) classifies chest X-rays as **pneumonia / normal**, and **Grad-CAM** shows
where in the image the model sees pneumonia. The model is trained only with image-level
labels (yes/no); no bounding-box labels are used for training. The boxes are used only to
measure localization quality (IoU).

The project follows the idea of the Stanford report *"Weakly Supervised Pneumonia
Localization"* (Huang, Monam, Cortes), using a stronger backbone (DenseNet121) and Grad-CAM.

**Team:** [Name 1], [Name 2], [Name 3]

---

## How it works

1. **Data processing:** DICOM X-ray -> rescale / fix inverted images -> 8-bit grayscale ->
   resize to 224x224 -> ImageNet mean/std normalization.
2. **Classifier:** ImageNet-pretrained DenseNet121 with a single-logit head (dropout 0.2).
   Trained with BCE loss and Adam (lr 1e-4, 15 epochs); the best checkpoint is chosen by
   validation ROC-AUC. Predicted pneumonia if probability >= 0.5.
3. **Grad-CAM:** heatmap of the regions that raised the pneumonia score (layer `norm5`
   or `denseblock3`).
4. **Localization:** heatmap thresholded -> connected regions -> bounding boxes, compared
   with radiologist boxes using IoU. The threshold is chosen on the validation set and the
   test set is used once.

## Repository structure

```
.
├── README.md
├── rc/                  # Python scripts: model, dataset, train, evaluate, gradcam, localization
├── notebooks/
│   └── Pneumonia_Detection_GradCAM.ipynb   # all-in-one notebook (demo + localization)
├── metadata/            # train.csv, val.csv, test.csv (+ annotations)
└── results/             # test metrics, confusion matrix, ROC curve, localization outputs
```

## Dataset

[RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- chest X-rays in DICOM format; label `Target` (0 = normal, 1 = pneumonia); boxes for pneumonia cases.
- The images are **not** stored in this repo. Download them from Kaggle (you must accept the
  competition rules). The train / validation / test splits are in `metadata/`.

## Trained model

Download the trained weights: **[ADD LINK HERE]** (Google Drive, "Anyone with the link can view").

Place the file at:
```
/content/drive/MyDrive/Pneumonia_Project/checkpoints/best_model.pth
```
(or change `CHECKPOINT_PATH` in the notebook's config cell). No training is needed.

## Setup and run (Google Colab)

1. Open `notebooks/Pneumonia_Detection_GradCAM.ipynb` in [Google Colab](https://colab.research.google.com).
2. (Optional) Runtime -> Change runtime type -> GPU. The demo also works on CPU.
3. Put the model file in your Google Drive (see above).
4. In the config cell (section 1), set `PROJECT_DIR` to your Drive folder and, for sections
   that need the data, `IMG_DIR` to the folder with the DICOM images.
5. Run the sections in this order:

   | Section | Run? | What it does |
   |---|---|---|
   | 1-5 | Yes | Setup, model, data code, Grad-CAM, load the trained model |
   | 6 | Skip | Optional test-set evaluation (`RUN_EVALUATION = False` by default) |
   | 7 | Yes | Prediction + Grad-CAM function |
   | 8 | Yes (needs images + `all_annotations.csv`) | Localization: IoU of Grad-CAM boxes vs radiologist boxes |
   | 9 | Yes | **Demo:** upload a chest X-ray (`.dcm`, `.png`, `.jpg`) and get the prediction, scores and Grad-CAM |

Python packages (preinstalled on Colab except `pydicom`):
```
pip install pydicom
# torch, torchvision, numpy, pandas, scikit-learn, matplotlib, scipy, pillow
```

### Running the scripts instead (optional)

```bash
python rc/train.py        # train (not needed; a trained model is provided)
python rc/evaluate.py     # evaluate on test.csv
python rc/localization.py # localization evaluation (needs gradcam.py = gradcam_corrected.py)
```
Default paths in the scripts point to `/content/drive/MyDrive/Pneumonia_Project/...`;
override them with the command-line arguments.

## Results

**Classification (held-out test set: 2,669 images, 601 pneumonia / 2,068 normal)**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.912 | 0.754 | 0.904 | 0.822 | 0.929 |

Confusion matrix `[[TN, FP], [FN, TP]]` = `[[1891, 177], [58, 543]]`.
See `results/` for `test_metrics.json`, `confusion_matrix.png` and `roc_curve.png`.

**Localization (IoU on test set)**

| Method | IoU |
|---|---|
| Ours: Grad-CAM boxes | [ADD] |
| Ours, paper-style (missed cases = 0) | [ADD] |
| Paper: CNN + CAM (weakly supervised) | 0.1508 |
| Paper: R-CNN (supervised) | 0.1266 |

## Limitations

- Boxes from Grad-CAM are approximate; IoU is low, as is typical for weak supervision.
- Heatmaps can land outside the lungs.
- Grad-CAM boxes are only meaningful when pneumonia is predicted.
- Educational project. **Not a medical device and not for diagnosis.**

## References

- Huang, Monam, Cortes. *Weakly Supervised Pneumonia Localization* (Stanford).
- Zhou et al. *Learning Deep Features for Discriminative Localization* (CAM), CVPR 2016.
- Selvaraju et al. *Grad-CAM*, ICCV 2017.
- Huang et al. *Densely Connected Convolutional Networks* (DenseNet), CVPR 2017.
