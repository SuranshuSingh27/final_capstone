"""
STEP 2 — PREPROCESSING PIPELINE
=================================
- Applies masks to RGB and MSI images
- Patient-level stratified 5-fold cross-validation splits
- Data augmentation (applied only on training fold)
- PyTorch Dataset classes for both RGB and MSI tracks
- Class weight computation for imbalance handling
- Saves fold splits to results/splits/

Usage:  python3 step2_preprocessing.py
Run this after step1_data_audit.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedGroupKFold

try:
    import spectral.io.envi as envi
except ImportError:
    print("ERROR: pip install spectral"); sys.exit(1)

# ─────────────────────────────────────────────
BASE_DIR      = Path("data/MODID")
PROCESSED_DIR = BASE_DIR / "processed"
RGB_DIR       = BASE_DIR / "rgb"
MASK_DIR      = BASE_DIR / "mask"
EXCEL_PATH    = BASE_DIR / "MODID_DESCRIPTOR.xlsx"
MANIFEST_PATH = Path("results/audit/manifest_clean.csv")
SPLITS_DIR    = Path("results/splits")
OUTPUT_DIR    = Path("results/preprocessing")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_SIZE = (224, 224)
MSI_SIZE = (128, 128)
N_FOLDS  = 5
SEED     = 42

LABEL_MAP = {
    "healthy":0,"control":0,"smoker":0,"osmf":0,
    "leukoplakia":0,"keratosis":0,"normal":0,"oscc":1,
}
CLASS_NAMES  = {0:"Non-Cancer", 1:"Cancer (OSCC)"}
CLASS_COLORS = {0:"#3B8BD4", 1:"#D85A30"}
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# SAFE LOADERS — handle any size or PIL issue
# ══════════════════════════════════════════════
def safe_load_rgb(path, fallback_shape=(270, 510, 3)):
    """Load RGB image. Returns black image on any failure."""
    try:
        return np.array(Image.open(str(path)).convert("RGB"))
    except Exception as e:
        print(f"  WARNING: RGB load failed {path} → {e}")
        return np.zeros(fallback_shape, dtype=np.uint8)


def safe_load_mask(path, fallback_shape=(270, 510)):
    """Load mask. Returns full-image mask on any failure."""
    try:
        return np.array(Image.open(str(path)).convert("L"))
    except Exception as e:
        print(f"  WARNING: mask load failed {path} → {e}")
        return np.ones(fallback_shape, dtype=np.uint8) * 255


# ══════════════════════════════════════════════
# LOAD MANIFEST
# ══════════════════════════════════════════════
def load_manifest():
    print("\n" + "="*55)
    print("  Loading manifest from audit step")
    print("="*55)

    if not MANIFEST_PATH.exists():
        print(f"  ERROR: {MANIFEST_PATH} not found.")
        print("  Run step1_data_audit.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["binary_label"].isin([0, 1])].reset_index(drop=True)
    df["image_id"]   = df["image_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)

    print(f"  Total usable images : {len(df)}")
    print(f"  Cancer              : {(df['binary_label']==1).sum()}")
    print(f"  Non-Cancer          : {(df['binary_label']==0).sum()}")
    print(f"  Unique patients     : {df['patient_id'].nunique()}")
    return df


# ══════════════════════════════════════════════
# PATIENT-LEVEL STRATIFIED K-FOLD
# ══════════════════════════════════════════════
def make_splits(df):
    print("\n" + "="*55)
    print(f"  Creating {N_FOLDS}-fold patient-level stratified splits")
    print("="*55)
    print("  NOTE: split by PATIENT — no data leakage")

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X      = np.arange(len(df))
    y      = df["binary_label"].values
    groups = df["patient_id"].values

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df   = df.iloc[train_idx].copy()
        val_df     = df.iloc[val_idx].copy()
        train_pats = set(train_df["patient_id"])
        val_pats   = set(val_df["patient_id"])
        overlap    = train_pats & val_pats
        assert len(overlap) == 0, f"Data leakage! {overlap}"

        folds.append({"train": train_df, "val": val_df})
        print(f"\n  Fold {fold_idx+1}:")
        print(f"    Train: {len(train_df):4d} images | "
              f"Cancer={(train_df['binary_label']==1).sum():3d} | "
              f"Non-Cancer={(train_df['binary_label']==0).sum():3d} | "
              f"Patients={len(train_pats):3d}")
        print(f"    Val  : {len(val_df):4d} images | "
              f"Cancer={(val_df['binary_label']==1).sum():3d} | "
              f"Non-Cancer={(val_df['binary_label']==0).sum():3d} | "
              f"Patients={len(val_pats):3d}")

        train_df.to_csv(SPLITS_DIR / f"fold{fold_idx+1}_train.csv", index=False)
        val_df.to_csv(SPLITS_DIR   / f"fold{fold_idx+1}_val.csv",   index=False)

    print(f"\n  Fold CSVs saved → {SPLITS_DIR}")
    return folds


# ══════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════
def compute_class_weights(df):
    n_total     = len(df)
    n_cancer    = (df["binary_label"] == 1).sum()
    n_noncancer = (df["binary_label"] == 0).sum()
    w0 = n_total / (2 * n_noncancer)
    w1 = n_total / (2 * n_cancer)
    print(f"\n  Class weights → Non-Cancer: {w0:.4f}  |  Cancer: {w1:.4f}")
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    torch.save(weights, SPLITS_DIR / "class_weights.pt")
    print(f"  Saved → {SPLITS_DIR / 'class_weights.pt'}")
    return weights


# ══════════════════════════════════════════════
# AUGMENTATION — RGB
# ══════════════════════════════════════════════
class RGBAugmentation:
    def __init__(self, size=RGB_SIZE):
        self.size = size

    def __call__(self, rgb_np, mask_np, training=True):
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
        img    = Image.fromarray(rgb_np)
        mask   = Image.fromarray((mask_np * 255).astype(np.uint8))

        img  = img.resize(self.size,  Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        if training:
            if np.random.rand() > 0.5:
                img  = TF.hflip(img);  mask = TF.hflip(mask)
            if np.random.rand() > 0.5:
                img  = TF.vflip(img);  mask = TF.vflip(mask)
            angle = np.random.uniform(-30, 30)
            img   = TF.rotate(img,  angle)
            mask  = TF.rotate(mask, angle)
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05)
            img = jitter(img)

        img_t  = TF.to_tensor(img)
        img_t  = TF.normalize(img_t,
                               mean=[0.485, 0.456, 0.406],
                               std= [0.229, 0.224, 0.225])
        mask_t = torch.from_numpy(np.array(mask) / 255.0).float()
        return img_t, mask_t


# ══════════════════════════════════════════════
# AUGMENTATION — MSI
# ══════════════════════════════════════════════
class MSIAugmentation:
    def __init__(self, size=MSI_SIZE):
        self.size = size

    def __call__(self, msi_np, mask_np, training=True):
        H, W, C = msi_np.shape
        mask    = Image.fromarray((mask_np * 255).astype(np.uint8))

        msi_resized = np.zeros((*self.size, C), dtype=np.float32)
        for b in range(C):
            band_img = Image.fromarray(msi_np[:, :, b], mode='F')
            msi_resized[:, :, b] = np.array(band_img.resize(self.size, Image.BILINEAR))
        mask = mask.resize(self.size, Image.NEAREST)

        if training:
            if np.random.rand() > 0.5:
                msi_resized = msi_resized[:, ::-1, :].copy()
                mask        = TF.hflip(mask)
            if np.random.rand() > 0.5:
                msi_resized = msi_resized[::-1, :, :].copy()
                mask        = TF.vflip(mask)
            angle   = np.random.uniform(-30, 30)
            rotated = np.zeros_like(msi_resized)
            for b in range(C):
                band_img = Image.fromarray(msi_resized[:, :, b], mode='F')
                rotated[:, :, b] = np.array(TF.rotate(band_img, angle))
            msi_resized = rotated
            mask        = TF.rotate(mask, angle)
            msi_resized = msi_resized + np.random.normal(
                0, 0.01, msi_resized.shape).astype(np.float32)

        # Per-band z-score
        msi_norm = np.zeros_like(msi_resized)
        for b in range(C):
            band = msi_resized[:, :, b]
            msi_norm[:, :, b] = (band - band.mean()) / (band.std() + 1e-8)

        msi_tensor  = torch.from_numpy(msi_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(np.array(mask) / 255.0).float()
        return msi_tensor, mask_tensor


# ══════════════════════════════════════════════
# PYTORCH DATASET — RGB
# ══════════════════════════════════════════════
class OralRGBDataset(Dataset):
    def __init__(self, df, training=True, size=RGB_SIZE):
        self.df       = df.reset_index(drop=True)
        self.training = training
        self.augment  = RGBAugmentation(size=size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        rgb_np   = safe_load_rgb(row["rgb_path"])
        mask_raw = safe_load_mask(row["mask_path"])

        # Align mask to image size
        if mask_raw.shape != rgb_np.shape[:2]:
            mask_raw = np.array(Image.fromarray(mask_raw).resize(
                (rgb_np.shape[1], rgb_np.shape[0]), Image.NEAREST))
        mask_np    = (mask_raw > 127).astype(np.uint8)
        rgb_masked = rgb_np * mask_np[..., None]

        img_tensor, mask_tensor = self.augment(rgb_masked, mask_np, self.training)
        label = torch.tensor(int(row["binary_label"]), dtype=torch.long)

        return {
            "image"    : img_tensor,
            "mask"     : mask_tensor,
            "label"    : label,
            "image_id" : str(row["image_id"]),
            "diag"     : str(row.get("diag_raw", "")),
        }


# ══════════════════════════════════════════════
# PYTORCH DATASET — MSI
# ══════════════════════════════════════════════
class OralMSIDataset(Dataset):
    def __init__(self, df, training=True, size=MSI_SIZE):
        self.df       = df.reset_index(drop=True)
        self.training = training
        self.augment  = MSIAugmentation(size=size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            hdr    = envi.open(str(row["hdr_path"]))
            msi_np = np.clip(hdr.load().astype(np.float32), 0, None)
        except Exception as e:
            print(f"  WARNING: MSI load failed {row['hdr_path']} → {e}")
            msi_np = np.zeros((270, 510, 16), dtype=np.float32)

        mask_raw = safe_load_mask(row["mask_path"])
        if mask_raw.shape != msi_np.shape[:2]:
            mask_raw = np.array(Image.fromarray(mask_raw).resize(
                (msi_np.shape[1], msi_np.shape[0]), Image.NEAREST))
        mask_np    = (mask_raw > 127).astype(np.uint8)
        msi_masked = msi_np * mask_np[..., None]

        msi_tensor, mask_tensor = self.augment(msi_masked, mask_np, self.training)
        label = torch.tensor(int(row["binary_label"]), dtype=torch.long)

        return {
            "image"    : msi_tensor,
            "mask"     : mask_tensor,
            "label"    : label,
            "image_id" : str(row["image_id"]),
            "diag"     : str(row.get("diag_raw", "")),
        }


# ══════════════════════════════════════════════
# SPECTRAL VECTOR DATASET
# ══════════════════════════════════════════════
class OralSpectralVectorDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            hdr    = envi.open(str(row["hdr_path"]))
            msi_np = np.clip(hdr.load().astype(np.float32), 0, None)
        except Exception:
            msi_np = np.zeros((270, 510, 16), dtype=np.float32)

        mask_raw = safe_load_mask(row["mask_path"])
        if mask_raw.shape != msi_np.shape[:2]:
            mask_raw = np.array(Image.fromarray(mask_raw).resize(
                (msi_np.shape[1], msi_np.shape[0]), Image.NEAREST))
        mask_np = mask_raw > 127
        roi     = msi_np[mask_np]
        vec     = roi.mean(axis=0) if len(roi) > 0 else msi_np.reshape(-1, 16).mean(axis=0)

        return {
            "vector"   : torch.from_numpy(vec).float(),
            "label"    : torch.tensor(int(row["binary_label"]), dtype=torch.long),
            "image_id" : str(row["image_id"]),
            "diag"     : str(row.get("diag_raw", "")),
        }


# ══════════════════════════════════════════════
# VERIFY DATASETS
# ══════════════════════════════════════════════
def verify_datasets(folds, df):
    print("\n" + "="*55)
    print("  Verifying datasets with fold 1 sample")
    print("="*55)

    train_df = folds[0]["train"]
    val_df   = folds[0]["val"]

    print("\n  --- RGB Dataset ---")
    rgb_train = OralRGBDataset(train_df, training=True)
    rgb_val   = OralRGBDataset(val_df,   training=False)
    sample    = rgb_train[0]
    print(f"  Train size  : {len(rgb_train)}")
    print(f"  Val size    : {len(rgb_val)}")
    print(f"  Image shape : {sample['image'].shape}  (expect 3×224×224)")
    print(f"  Mask  shape : {sample['mask'].shape}   (expect 224×224)")
    print(f"  Label       : {sample['label'].item()}  ({CLASS_NAMES[sample['label'].item()]})")
    print(f"  Image range : [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")

    print("\n  --- MSI Dataset ---")
    msi_train = OralMSIDataset(train_df, training=True)
    msi_val   = OralMSIDataset(val_df,   training=False)
    sample_m  = msi_train[0]
    print(f"  Train size  : {len(msi_train)}")
    print(f"  Val size    : {len(msi_val)}")
    print(f"  Image shape : {sample_m['image'].shape}  (expect 16×128×128)")
    print(f"  Mask  shape : {sample_m['mask'].shape}   (expect 128×128)")
    print(f"  Label       : {sample_m['label'].item()}  ({CLASS_NAMES[sample_m['label'].item()]})")

    print("\n  --- Spectral Vector Dataset ---")
    vec_ds   = OralSpectralVectorDataset(train_df)
    sample_v = vec_ds[0]
    print(f"  Vector shape: {sample_v['vector'].shape}  (expect 16)")
    print(f"  Label       : {sample_v['label'].item()}")
    print("\n  ✓ All datasets verified successfully")


# ══════════════════════════════════════════════
# VISUALIZE AUGMENTATION
# ══════════════════════════════════════════════
def visualize_augmentation(df):
    print("\n  Visualizing augmentation effect...")

    sample_row = df[df["binary_label"] == 1].iloc[0]
    rgb_np     = safe_load_rgb(sample_row["rgb_path"])
    mask_raw   = safe_load_mask(sample_row["mask_path"])
    mask_np    = (mask_raw > 127).astype(np.uint8)
    rgb_masked = rgb_np * mask_np[..., None]

    aug  = RGBAugmentation()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Augmentation samples — OSCC (Cancer)",
                 fontsize=12, fontweight="bold")

    orig_t, _ = aug(rgb_masked, mask_np, training=False)
    orig_disp = ((orig_t*std+mean).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    axes[0][0].imshow(orig_disp); axes[0][0].set_title("Original"); axes[0][0].axis("off")

    for i in range(1, 8):
        ax      = axes[i//4][i%4]
        aug_t,_ = aug(rgb_masked, mask_np, training=True)
        disp    = ((aug_t*std+mean).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        ax.imshow(disp); ax.set_title(f"Aug #{i}"); ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "augmentation_samples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  STEP 2 — PREPROCESSING PIPELINE")
    print("█"*55)

    df      = load_manifest()
    folds   = make_splits(df)
    weights = compute_class_weights(df)

    print("\n" + "="*55)
    print("  Class weights for training:")
    print(f"    Non-Cancer : {weights[0]:.4f}")
    print(f"    Cancer     : {weights[1]:.4f}")
    print("="*55)

    verify_datasets(folds, df)
    visualize_augmentation(df)

    config = {
        "n_folds"         : N_FOLDS,
        "seed"            : SEED,
        "rgb_size"        : list(RGB_SIZE),
        "msi_size"        : list(MSI_SIZE),
        "n_bands"         : 16,
        "n_classes"       : 2,
        "class_names"     : CLASS_NAMES,
        "class_weights"   : weights.tolist(),
        "total_images"    : len(df),
        "cancer_images"   : int((df["binary_label"]==1).sum()),
        "noncancer_images": int((df["binary_label"]==0).sum()),
    }
    with open(SPLITS_DIR / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved → {SPLITS_DIR / 'dataset_config.json'}")

    print("\n" + "="*55)
    print("  STEP 2 COMPLETE")
    print(f"  Fold CSVs → {SPLITS_DIR}")
    print(f"  Plots     → {OUTPUT_DIR}")
    print("  Next step → step3_rgb_models.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
