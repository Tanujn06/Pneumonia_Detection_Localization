
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_xray(image_path):
    """Read a DICOM X-ray and convert it to an 8-bit grayscale image."""
    dicom = pydicom.dcmread(str(image_path))
    image = dicom.pixel_array.astype(np.float32)

    # Apply DICOM rescale parameters when available.
    slope = float(getattr(dicom, "RescaleSlope", 1))
    intercept = float(getattr(dicom, "RescaleIntercept", 0))
    image = image * slope + intercept

    # Handle MONOCHROME1 images, where low values are displayed as white.
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    # Normalize pixel intensities to 0–255.
    image -= image.min()
    if image.max() > 0:
        image = image / image.max() * 255.0

    return image.astype(np.uint8)


class PneumoniaDataset(Dataset):
    """Dataset for binary pneumonia classification."""

    def __init__(
        self,
        metadata,
        image_dir,
        transform=None,
        patient_col="patientId",
        label_col="Target",
    ):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform or T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

        self.patient_col = patient_col
        self.label_col = label_col

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        patient_id = str(row[self.patient_col])
        image_path = self.image_dir / f"{patient_id}.dcm"

        if not image_path.exists():
            raise FileNotFoundError(
                f"X-ray not found: {image_path}"
            )

        image = Image.fromarray(read_xray(image_path)).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            float(row[self.label_col]),
            dtype=torch.float32,
        )

        return image, label, patient_id

