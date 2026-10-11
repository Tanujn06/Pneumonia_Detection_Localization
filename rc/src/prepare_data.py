
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# Adjust these paths to match your project.
DATA_DIR = Path("data")
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
OUTPUT_DIR = DATA_DIR / "metadata"


def prepare_splits(
    labels_csv=LABELS_CSV,
    output_dir=OUTPUT_DIR,
):
    labels_csv = Path(labels_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_csv}\n"
            "Check that the CSV is available at this path."
        )

    # Preserve every original annotation, including bounding boxes.
    annotations = pd.read_csv(labels_csv)

    required_columns = {"patientId", "Target"}
    missing = required_columns - set(annotations.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    annotations["Target"] = annotations["Target"].astype(int)

    if not annotations["Target"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 or 1.")

    # Save a complete copy for localization.
    annotations.to_csv(
        output_dir / "all_annotations.csv",
        index=False,
    )

    # Create one classification label per patient.
    # Multiple positive bounding-box rows become one positive patient.
    patient_labels = (
        annotations.groupby("patientId", as_index=False)["Target"]
        .max()
    )

    # Split patients, not annotation rows, to prevent leakage.
    train_df, remaining_df = train_test_split(
        patient_labels,
        test_size=0.20,
        stratify=patient_labels["Target"],
        random_state=RANDOM_STATE,
    )

    # Divide the remaining 20% equally into validation and test.
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["Target"],
        random_state=RANDOM_STATE,
    )

    splits = {
        "train.csv": train_df,
        "val.csv": val_df,
        "test.csv": test_df,
    }

    for filename, split_df in splits.items():
        split_df = split_df.reset_index(drop=True)
        split_df.to_csv(output_dir / filename, index=False)

        print(f"\n{filename}: {len(split_df)} patients")
        print(
            "Class counts:",
            split_df["Target"].value_counts().sort_index().to_dict(),
        )

    # Verify patient IDs do not overlap between splits.
    train_ids = set(train_df["patientId"])
    val_ids = set(val_df["patientId"])
    test_ids = set(test_df["patientId"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nPatient-level split verification passed.")
    print("Total unique patients:", len(patient_labels))
    print(
        "Original annotation rows preserved:",
        len(annotations),
    )
    print("Output directory:", output_dir.resolve())


if __name__ == "__main__":
    prepare_splits()
