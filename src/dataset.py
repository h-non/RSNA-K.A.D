


import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ── Paths ─────────────────────────────────────────────────────────────────────
MRI_DIR        = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series")
LABELS_CSV     = Path("/kaggle/input/datasets/nikolozkozmanashvili/softlabelsdb/soft_labels_final.csv")  # adjust
DINOV2_PATH    = Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1")
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
PREPROCESS_DIR = Path("/kaggle/input/datasets/nikolozkozmanashvili/kadmris5/preprocessed")


# ── Config ────────────────────────────────────────────────────────────────────
CONDITIONS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
]

PLANES      = ["Sagittal", "Coronal", "Axial"]
N_SLICES    = 5        # slices per plane (2.5D — center 5)
IMG_SIZE    = 224      # DINOv2 native input size
BATCH_SIZE  = 4
EPOCHS      = 50
LR          = 3e-5
NUM_LABELS  = 12

# ── GPU ───────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


##--------------------
import pandas as pd

series_df = pd.read_csv("/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series.csv")

def select_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one series per study per plane using fluid-sensitive priority."""
    selected = (
        series_df
        .sort_values(["Fluid_Sensitive", "Fat_Suppression"], ascending=False)
        .groupby(["StudyInstanceUID", "Anatomical_Plane"])
        .first()
        .reset_index()
    )
    return selected[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]]

series_map = select_series(series_df)
print(f"series_map ready: {len(series_map)} series across {series_map['StudyInstanceUID'].nunique()} studies")

##--------------------

import pydicom
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


#MAY BE MISSING IMPORTS


PLANES     = ["Sagittal", "Coronal", "Axial"]
N_SLICES   = 5      # center 5 slices per plane (2.5D)
IMG_SIZE   = 224    # DINOv2 native size

# ── DICOM helpers ─────────────────────────────────────────────────────────────
def load_dicom_volume(series_path: Path) -> np.ndarray:
    """Load all slices in a series, sort by InstanceNumber, return (N, H, W) uint8."""
    dcm_files = sorted(series_path.glob("*.dcm"))
    slices = []
    for f in dcm_files:
        ds  = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        # Normalize to 0-255
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        slices.append(img)
    return np.stack(slices, axis=0)   # (N, H, W)


def select_center_slices(volume: np.ndarray, n: int = N_SLICES) -> np.ndarray:
    N     = volume.shape[0]
    mid   = N // 2
    half  = n // 2
    start = max(0, mid - half)
    end   = min(N, start + n)
    slices = volume[start:end]
    # Pad with edge slice if volume has fewer than n slices
    if len(slices) < n:
        pad    = np.repeat(slices[-1:], n - len(slices), axis=0)
        slices = np.concatenate([slices, pad], axis=0)
    return slices  # always (n, H, W)

def preprocess_slice(img: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    """Resize, convert to 3-channel, normalize for DINOv2."""
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=2)   # (H, W, 3) grayscale→RGB
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = T.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(img)
    return img   # (5, H, W)

def load_study_tensor(study_uid: str, series_map: pd.DataFrame,
                      mri_dir: Path) -> torch.Tensor:
    """
    Load all 3 planes for a study.
    Returns tensor of shape (3*N_SLICES, 3, H, W) = (9, 3, 224, 224)
    """
    study_series = series_map[series_map["StudyInstanceUID"] == study_uid]
    plane_tensors = []

    for plane in PLANES:
        row = study_series[study_series["Anatomical_Plane"] == plane]
        if len(row) == 0:
            # Missing plane — fill with zeros
            plane_tensors.append(torch.zeros(N_SLICES, 3, IMG_SIZE, IMG_SIZE))
            continue

        series_uid  = row.iloc[0]["SeriesInstanceUID"]
        series_path = mri_dir / study_uid / series_uid

        volume  = load_dicom_volume(series_path)          # (N, H, W)
        slices  = select_center_slices(volume, N_SLICES)  # (3, H, W)
        tensors = torch.stack([preprocess_slice(s) for s in slices])  # (3, 3, 224, 224)
        plane_tensors.append(tensors)

    return torch.cat(plane_tensors, dim=0)   # (9, 3, 224, 224)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KneeDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, series_map: pd.DataFrame,
                 mri_dir: Path, augment: bool = False):
        self.labels_df  = labels_df.reset_index(drop=True)
        self.series_map = series_map
        self.mri_dir    = mri_dir
        self.augment    = augment

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row       = self.labels_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        labels    = torch.tensor(row[CONDITIONS].values.astype(np.float32))

        images = load_study_tensor(study_uid, self.series_map, self.mri_dir)

        if self.augment:
            do_flip  = random.random() > 0.5
            angle    = random.uniform(-15, 15)
            

            def augment_slice(img):
                if do_flip:
                    img = TF.hflip(img)
                img = TF.rotate(img, angle)
                return img

            images = torch.stack([augment_slice(img) for img in images])

        return images, labels, study_uid


##--------------------

# Quick sanity check on one study
import pandas as pd

labels_df = pd.read_csv("/kaggle/input/datasets/nikolozkozmanashvili/softlabelsdb/soft_labels_final.csv")
labels_df = labels_df.rename(columns={"Baker's": "Bakers"})  # ← add this line

test_uid  = labels_df["StudyInstanceUID"].iloc[0]
tensor    = load_study_tensor(test_uid, series_map, MRI_DIR)

print(f"Output shape : {tensor.shape}")
print(f"Value range  : {tensor.min():.2f} to {tensor.max():.2f}")
print(f"Dtype        : {tensor.dtype}")


#---------------------------------------------------------------



class KneeDatasetFast(Dataset):
    def __init__(self, labels_df: pd.DataFrame, preprocess_dir: Path,
                 augment: bool = False):
        self.labels_df      = labels_df.reset_index(drop=True)
        self.preprocess_dir = preprocess_dir
        self.augment        = augment

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row       = self.labels_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        labels    = torch.tensor(row[CONDITIONS].values.astype(np.float32))

        plane_tensors = []
        for plane in PLANES:
            plane_dir = self.preprocess_dir / study_uid / plane
            slices    = []
            for i in range(N_SLICES):
                png_path = plane_dir / f"slice_{i}.png"
                if png_path.exists():
                    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                else:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                slices.append(preprocess_slice(img))
            plane_tensors.append(torch.stack(slices))

        images = torch.cat(plane_tensors, dim=0)

        if self.augment:
            do_flip  = random.random() > 0.5
            angle    = random.uniform(-15, 15)
           

            def augment_slice(img):
                if do_flip:
                    img = TF.hflip(img)
                img = TF.rotate(img, angle)

                return img

            images = torch.stack([augment_slice(img) for img in images])

        return images, labels, study_uid


train_ds_fast = KneeDatasetFast(train_labels, PREPROCESS_DIR, augment=True)
val_ds_fast   = KneeDatasetFast(val_labels,   PREPROCESS_DIR, augment=False)

train_loader  = DataLoader(train_ds_fast, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_ds_fast,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds_fast)} | {len(train_loader)} batches")
print(f"Val  : {len(val_ds_fast)}  | {len(val_loader)} batches")