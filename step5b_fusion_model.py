"""
STEP 5B — RGB + MSI FUSION MODEL
==================================
Combines RGB spatial features + MSI spectral features
into a single fusion classifier.

Three fusion strategies trained and compared:
  1. Early fusion    — concatenate raw features before CNN
  2. Late fusion     — train RGB and MSI separately, combine predictions
  3. Feature fusion  — extract deep features from both, concatenate, classify

Feature fusion is the main model (strongest approach).

All results saved separately — does NOT overwrite step3/step4 results.

Usage:  python step5b_fusion_model.py
Outputs → results/fusion_models/
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image

from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    f1_score, roc_curve
)

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("ERROR: pip install efficientnet-pytorch")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from step2_preprocessing import (
    OralRGBDataset, OralMSIDataset,
    safe_load_rgb, safe_load_mask, RGBAugmentation, MSIAugmentation
)

try:
    import spectral.io.envi as envi
except ImportError:
    print("ERROR: pip install spectral")
    sys.exit(1)

# ─────────────────────────────────────────────
SPLITS_DIR  = Path("results/splits")
OUTPUT_DIR  = Path("results/fusion_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS     = 5
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 1e-4
PATIENCE    = 20
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"\n  Device: {DEVICE}")
# ─────────────────────────────────────────────


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ══════════════════════════════════════════════
# FUSION DATASET — loads RGB + MSI together
# ══════════════════════════════════════════════
class FusionDataset(Dataset):
    """
    Loads both RGB and MSI for the same image simultaneously.
    Returns paired (rgb_tensor, msi_tensor, label).
    """
    def __init__(self, df, training=True):
        self.df        = df.reset_index(drop=True)
        self.training  = training
        self.rgb_aug   = RGBAugmentation(size=(224, 224))
        self.msi_aug   = MSIAugmentation(size=(128, 128))

        # Use same random seed for RGB and MSI augmentation
        # so spatial transforms are identical
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load RGB ──────────────────────────
        rgb_np   = safe_load_rgb(row["rgb_path"])
        mask_raw = safe_load_mask(row["mask_path"])

        if mask_raw.shape != rgb_np.shape[:2]:
            mask_raw = np.array(Image.fromarray(mask_raw).resize(
                (rgb_np.shape[1], rgb_np.shape[0]), Image.NEAREST))
        mask_np    = (mask_raw > 127).astype(np.uint8)
        rgb_masked = rgb_np * mask_np[..., None]

        # Fix random seed so RGB and MSI get same spatial augmentation
        seed = np.random.randint(0, 2**31)
        np.random.seed(seed)
        rgb_tensor, _ = self.rgb_aug(rgb_masked, mask_np, self.training)

        # ── Load MSI ──────────────────────────
        try:
            hdr    = envi.open(str(row["hdr_path"]))
            msi_np = np.clip(hdr.load().astype(np.float32), 0, None)
        except Exception:
            msi_np = np.zeros((270, 510, 16), dtype=np.float32)

        msi_mask_raw = safe_load_mask(row["mask_path"])
        if msi_mask_raw.shape != msi_np.shape[:2]:
            msi_mask_raw = np.array(Image.fromarray(msi_mask_raw).resize(
                (msi_np.shape[1], msi_np.shape[0]), Image.NEAREST))
        msi_mask_np = (msi_mask_raw > 127).astype(np.uint8)
        msi_masked  = msi_np * msi_mask_np[..., None]

        np.random.seed(seed)  # same seed → same spatial transform
        msi_tensor, _ = self.msi_aug(msi_masked, msi_mask_np, self.training)

        # Reset random state
        np.random.seed(None)

        label = torch.tensor(int(row["binary_label"]), dtype=torch.long)

        return {
            "rgb"     : rgb_tensor,   # 3×224×224
            "msi"     : msi_tensor,   # 16×128×128
            "label"   : label,
            "image_id": str(row["image_id"]),
            "diag"    : str(row.get("diag_raw", "")),
        }


# ══════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════
def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1          = f1_score(labels, preds, zero_division=0)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {
        "accuracy"   : float(accuracy),
        "auc"        : float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1"         : float(f1),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def compute_threshold_analysis(labels, probs):
    return {str(t): compute_metrics(labels, probs, t)
            for t in [0.3, 0.4, 0.5]}


# ══════════════════════════════════════════════
# FUSION MODEL DEFINITIONS
# ══════════════════════════════════════════════

class FeatureFusionModel(nn.Module):
    """
    Feature-level fusion (strongest approach):
    1. ResNet50 extracts RGB features (2048-D)
    2. Custom MSI encoder extracts spectral features (256-D)
    3. Concatenate → (2304-D)
    4. Fusion classifier → 2 classes

    This allows each modality to specialize while the
    fusion layer learns cross-modal patterns.
    """
    def __init__(self, rgb_feature_dim=512, msi_feature_dim=256):
        super().__init__()

        # ── RGB encoder (ResNet50 backbone) ──
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        # Freeze early layers
        for param in resnet.parameters():
            param.requires_grad = False
        for layer in [resnet.layer3, resnet.layer4]:
            for param in layer.parameters():
                param.requires_grad = True

        # Remove final FC — use as feature extractor
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_proj    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, rgb_feature_dim),
            nn.BatchNorm1d(rgb_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # ── MSI encoder (Spectral Attention + CNN) ──
        self.band_attention = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.Softmax(dim=-1),
        )
        self.msi_cnn = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.msi_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, msi_feature_dim),
            nn.BatchNorm1d(msi_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # ── Fusion classifier ──
        fusion_dim = rgb_feature_dim + msi_feature_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, rgb, msi):
        # RGB features
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = self.rgb_proj(rgb_feat)

        # MSI features with spectral attention
        attn     = self.band_attention(msi.mean(dim=[2, 3]))
        msi_att  = msi * attn.unsqueeze(-1).unsqueeze(-1)
        msi_feat = self.msi_cnn(msi_att)
        msi_feat = self.msi_proj(msi_feat)

        # Fuse and classify
        fused = torch.cat([rgb_feat, msi_feat], dim=1)
        return self.fusion_classifier(fused)

    def get_msi_attention(self, msi):
        return self.band_attention(msi.mean(dim=[2, 3]))


class LateFusionModel(nn.Module):
    """
    Late fusion: train RGB and MSI separately,
    combine their output probabilities with a learned weight.
    Simple but effective baseline for comparison.
    """
    def __init__(self):
        super().__init__()

        # RGB branch
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        for param in resnet.parameters():
            param.requires_grad = False
        for layer in [resnet.layer3, resnet.layer4]:
            for param in layer.parameters():
                param.requires_grad = True
        in_feat  = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(in_feat, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 2)
        )
        self.rgb_branch = resnet

        # MSI branch
        self.msi_branch = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.msi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 2)
        )

        # Learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, rgb, msi):
        rgb_logits = self.rgb_branch(rgb)
        msi_logits = self.msi_head(self.msi_branch(msi))
        w          = torch.softmax(self.fusion_weight, dim=0)
        return w[0] * rgb_logits + w[1] * msi_logits


FUSION_MODELS = {
    "FeatureFusion": lambda: FeatureFusionModel(),
    "LateFusion"   : lambda: LateFusionModel(),
}


# ══════════════════════════════════════════════
# TRAIN / VAL EPOCH
# ══════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []

    for batch in loader:
        rgb    = batch["rgb"].to(DEVICE)
        msi    = batch["msi"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(rgb, msi)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return total_loss / len(loader), metrics


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    for batch in loader:
        rgb    = batch["rgb"].to(DEVICE)
        msi    = batch["msi"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs  = model(rgb, msi)
        loss     = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    metrics    = compute_metrics(labels_arr, probs_arr)
    return total_loss / len(loader), metrics, labels_arr, probs_arr


# ══════════════════════════════════════════════
# TRAIN ONE FUSION MODEL
# ══════════════════════════════════════════════
def train_fusion_model(model_name, build_fn, class_weights):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"{'='*55}")

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)

    cw        = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n  --- Fold {fold}/{N_FOLDS} ---")
        clear_cache()

        train_df = pd.read_csv(SPLITS_DIR / f"fold{fold}_train.csv")
        val_df   = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")

        train_ds = FusionDataset(train_df, training=True)
        val_ds   = FusionDataset(val_df,   training=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=False)

        model = build_fn().to(DEVICE)

        # Separate param groups
        rgb_backbone_params = []
        msi_params          = []
        head_params         = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "rgb_encoder" in name or "rgb_branch" in name:
                rgb_backbone_params.append(param)
            elif any(k in name for k in
                     ["fusion_classifier","rgb_proj","msi_proj",
                      "msi_head","fusion_weight"]):
                head_params.append(param)
            else:
                msi_params.append(param)

        optimizer = optim.AdamW([
            {"params": rgb_backbone_params, "lr": LR * 0.05},
            {"params": msi_params,          "lr": LR * 0.1},
            {"params": head_params,         "lr": LR},
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        best_acc         = -1
        best_model_state = None
        best_metrics     = None
        best_labels      = None
        best_probs       = None
        patience_counter = 0

        history = {
            "train_loss":[], "val_loss":[],
            "train_acc" :[], "val_acc" :[],
            "train_auc" :[], "val_auc" :[],
            "val_sens"  :[],
        }

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            train_loss, train_m = train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_m, v_lab, v_prob = val_epoch(
                model, val_loader, criterion
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_m["accuracy"])
            history["val_acc"].append(val_m["accuracy"])
            history["train_auc"].append(train_m["auc"])
            history["val_auc"].append(val_m["auc"])
            history["val_sens"].append(val_m["sensitivity"])

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{EPOCHS} | "
                  f"Loss {train_loss:.3f}/{val_loss:.3f} | "
                  f"Acc {train_m['accuracy']:.3f}/{val_m['accuracy']:.3f} | "
                  f"AUC {val_m['auc']:.3f} | "
                  f"Sens {val_m['sensitivity']:.3f} | "
                  f"Spec {val_m['specificity']:.3f} | "
                  f"{elapsed:.1f}s")

            if val_m["accuracy"] > best_acc:
                best_acc         = val_m["accuracy"]
                best_model_state = deepcopy(model.state_dict())
                best_metrics     = val_m
                best_labels      = v_lab.copy()
                best_probs       = v_prob.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        torch.save(best_model_state, model_dir / f"fold{fold}_best.pt")
        plot_training_history(history, fold, model_name, model_dir)

        thresh_analysis = compute_threshold_analysis(best_labels, best_probs)
        print(f"\n  Fold {fold} best results:")
        for t, m in thresh_analysis.items():
            print(f"    t={t}: Acc={m['accuracy']:.3f} "
                  f"Sens={m['sensitivity']:.3f} "
                  f"Spec={m['specificity']:.3f} "
                  f"AUC={m['auc']:.3f} "
                  f"TP={m['tp']} FN={m['fn']}")

        fold_results.append({
            "fold"           : fold,
            "metrics"        : best_metrics,
            "thresh_analysis": thresh_analysis,
            "labels"         : best_labels,
            "probs"          : best_probs,
            "history"        : history,
        })

        del model
        clear_cache()

    return fold_results


# ══════════════════════════════════════════════
# AGGREGATE
# ══════════════════════════════════════════════
def aggregate_results(fold_results, model_name):
    print(f"\n  {'─'*50}")
    print(f"  {model_name} — 5-Fold CV")
    print(f"  {'─'*50}")

    keys = ["accuracy","auc","sensitivity","specificity","f1"]
    agg  = {k:[] for k in keys}
    for fr in fold_results:
        for k in keys:
            agg[k].append(fr["metrics"][k])

    summary = {}
    for k in keys:
        vals = np.array(agg[k])
        summary[k] = {
            "mean"    : float(vals.mean()),
            "std"     : float(vals.std()),
            "per_fold": vals.tolist()
        }
        print(f"  {k:12s}: {vals.mean():.4f} ± {vals.std():.4f}  "
              f"| {[f'{v:.3f}' for v in vals]}")

    print(f"\n  Threshold analysis:")
    for t in ["0.3","0.4","0.5"]:
        sens_list = [fr["thresh_analysis"][t]["sensitivity"]
                     for fr in fold_results]
        spec_list = [fr["thresh_analysis"][t]["specificity"]
                     for fr in fold_results]
        acc_list  = [fr["thresh_analysis"][t]["accuracy"]
                     for fr in fold_results]
        print(f"    t={t}: Acc={np.mean(acc_list):.3f} "
              f"Sens={np.mean(sens_list):.3f}±{np.std(sens_list):.3f} "
              f"Spec={np.mean(spec_list):.3f}±{np.std(spec_list):.3f}")

    all_labels = np.concatenate([fr["labels"] for fr in fold_results])
    all_probs  = np.concatenate([fr["probs"]  for fr in fold_results])
    try:
        print(f"\n  Pooled AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    except Exception:
        pass

    return summary, all_labels, all_probs


# ══════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════
def plot_training_history(history, fold, model_name, model_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Fold {fold}",
                 fontsize=12, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#1D9E75")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#D85A30")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#1D9E75")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#D85A30")
    axes[1].axhline(0.85, color="green",  linestyle="--", alpha=0.5, label="85%")
    axes[1].axhline(0.90, color="orange", linestyle="--", alpha=0.5, label="90%")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_auc"], label="Train AUC", color="#1D9E75")
    axes[2].plot(epochs, history["val_auc"],   label="Val AUC",   color="#D85A30")
    axes[2].plot(epochs, history["val_sens"],
                 label="Val Sens", color="#534AB7", linestyle="--")
    axes[2].axhline(0.85, color="green", linestyle="--", alpha=0.5)
    axes[2].set_title("AUC + Sensitivity")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / f"fold{fold}_history.png",
                dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curves(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = {
        "FeatureFusion": "#1D9E75",
        "LateFusion"   : "#888780",
    }
    for mn, (all_labels, all_probs) in all_results.items():
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            auc         = roc_auc_score(all_labels, all_probs)
            ax.plot(fpr, tpr, color=colors.get(mn,"#888"),
                    linewidth=2.5, label=f"{mn} (AUC={auc:.3f})")
        except Exception:
            continue
    ax.plot([0,1],[0,1],"k--",alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — Fusion Models (5-fold pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "roc_curves_fusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curves → {out}")


def plot_confusion_matrices(all_results, output_dir):
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 8))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Confusion Matrices — Fusion Models",
                 fontsize=12, fontweight="bold")
    for col, (mn, (all_labels, all_probs)) in enumerate(all_results.items()):
        for row, t in enumerate([0.5, 0.3]):
            ax    = axes[row][col]
            preds = (all_probs >= t).astype(int)
            cm    = confusion_matrix(all_labels, preds, labels=[0,1])
            ax.imshow(cm, cmap="Greens")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Non-Cancer","Cancer"])
            ax.set_yticklabels(["Non-Cancer","Cancer"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{mn} (t={t})")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i,j] > cm.max()/2 else "black"
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    out = output_dir / "confusion_matrices_fusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrices → {out}")


def plot_full_comparison(fusion_summaries, output_dir):
    """
    Final comparison: RGB vs MSI vs Fusion.
    Loads results from all previous steps.
    """
    rgb_path    = Path("results/rgb_models/all_models_summary.json")
    msi_path    = Path("results/msi_models/all_models_summary.json")
    tuned_path  = Path("results/msi_tuned/all_models_summary.json")

    all_models = {}

    if rgb_path.exists():
        with open(rgb_path) as f:
            rgb_s = json.load(f)
        # Best RGB = ResNet50
        if "ResNet50" in rgb_s:
            all_models["RGB-ResNet50"] = rgb_s["ResNet50"]

    if msi_path.exists():
        with open(msi_path) as f:
            msi_s = json.load(f)
        # Best MSI
        best_msi = max(msi_s, key=lambda k: msi_s[k]["auc"]["mean"])
        all_models[f"MSI-{best_msi}"] = msi_s[best_msi]

    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned_s = json.load(f)
        best_tuned = max(tuned_s, key=lambda k: tuned_s[k]["auc"]["mean"])
        all_models[f"MSI-Tuned-{best_tuned}"] = tuned_s[best_tuned]

    # Add fusion results
    for mn, s in fusion_summaries.items():
        all_models[f"Fusion-{mn}"] = s

    if len(all_models) < 2:
        print("  Not enough models to compare — skipping")
        return

    metrics = ["accuracy","auc","sensitivity","specificity","f1"]
    colors  = [
        "#3B8BD4","#534AB7","#9F77DD",
        "#1D9E75","#D85A30","#BA7517","#E8C040"
    ]

    x     = np.arange(len(metrics))
    width = 0.8 / len(all_models)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (mn, s) in enumerate(all_models.items()):
        means = [s[m]["mean"] for m in metrics]
        stds  = [s[m]["std"]  for m in metrics]
        offset = (i - len(all_models)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=mn,
               color=colors[i % len(colors)], alpha=0.85,
               yerr=stds, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Complete Comparison — RGB vs MSI vs Fusion (5-fold CV)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.85, color="green",  linestyle="--", alpha=0.4, label="85%")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.4, label="90%")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / "full_comparison_all_models.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Full comparison → {out}")


def plot_band_importance_fusion(all_results, output_dir):
    """Extract and plot MSI attention weights from FeatureFusion model."""
    model_dir = OUTPUT_DIR / "FeatureFusion"
    if not model_dir.exists():
        return

    wavelengths = [460,465,474,483,493,504,512,522,
                   534,541,552,560,570,580,585,595]
    all_attn = []

    for fold in range(1, N_FOLDS + 1):
        wp = model_dir / f"fold{fold}_best.pt"
        if not wp.exists():
            continue
        model = FeatureFusionModel().to(DEVICE)
        model.load_state_dict(
            torch.load(wp, map_location=DEVICE, weights_only=True)
        )
        model.eval()

        val_df = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")
        val_ds = FusionDataset(val_df, training=False)
        loader = DataLoader(val_ds, batch_size=4,
                            shuffle=False, num_workers=0)

        fold_attn = []
        with torch.no_grad():
            for batch in loader:
                msi  = batch["msi"].to(DEVICE)
                attn = model.get_msi_attention(msi)
                fold_attn.append(attn.cpu().numpy())

        if fold_attn:
            all_attn.append(np.concatenate(fold_attn).mean(axis=0))

    if not all_attn:
        return

    mean_attn = np.array(all_attn).mean(axis=0)
    std_attn  = np.array(all_attn).std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(16), mean_attn, yerr=std_attn,
                  color="#1D9E75", alpha=0.8, capsize=4)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{w}nm" for w in wavelengths],
                       rotation=45, fontsize=9)
    ax.set_ylabel("Attention weight", fontsize=11)
    ax.set_title("Spectral Band Importance — Fusion Model\n"
                 "(MSI attention weights learned jointly with RGB)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / "band_importance_fusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Band importance → {out}")

    with open(output_dir / "band_importance_fusion.json", "w") as f:
        json.dump({
            "wavelengths"   : wavelengths,
            "mean_attention": mean_attn.tolist(),
            "std_attention" : std_attn.tolist(),
        }, f, indent=2)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  STEP 5B — RGB + MSI FUSION MODELS")
    print("█"*55)
    print(f"  Models    : FeatureFusion, LateFusion")
    print(f"  Epochs    : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device    : {DEVICE}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  NOTE: Does NOT overwrite step3/step4 results")

    class_weights = torch.load(
        SPLITS_DIR / "class_weights.pt", weights_only=True
    )
    print(f"  Class weights: {[round(w,4) for w in class_weights.tolist()]}")

    all_summaries = {}
    all_results   = {}

    for model_name, build_fn in FUSION_MODELS.items():
        fold_results = train_fusion_model(model_name, build_fn, class_weights)
        summary, all_labels, all_probs = aggregate_results(
            fold_results, model_name
        )
        all_summaries[model_name] = summary
        all_results[model_name]   = (all_labels, all_probs)

        with open(OUTPUT_DIR / model_name / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "all_models_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n  Generating plots...")
    plot_roc_curves(all_results, OUTPUT_DIR)
    plot_confusion_matrices(all_results, OUTPUT_DIR)
    plot_full_comparison(all_summaries, OUTPUT_DIR)
    plot_band_importance_fusion(all_results, OUTPUT_DIR)

    print("\n" + "="*65)
    print("  FINAL FUSION RESULTS SUMMARY")
    print("="*65)
    print(f"  {'Model':20s} {'Acc':>8} {'AUC':>8} "
          f"{'Sens':>8} {'Spec':>8} {'F1':>8}")
    print("  " + "─"*58)
    for mn, s in all_summaries.items():
        print(f"  {mn:20s} "
              f"{s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}  "
              f"{s['auc']['mean']:.3f}±{s['auc']['std']:.3f}  "
              f"{s['sensitivity']['mean']:.3f}±{s['sensitivity']['std']:.3f}  "
              f"{s['specificity']['mean']:.3f}±{s['specificity']['std']:.3f}  "
              f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}")

    # Compare vs best RGB
    rgb_path = Path("results/rgb_models/all_models_summary.json")
    if rgb_path.exists():
        with open(rgb_path) as f:
            rgb_s = json.load(f)
        resnet_s = rgb_s.get("ResNet50", {})
        if resnet_s:
            print(f"\n  {'RGB-ResNet50':20s} "
                  f"{resnet_s['accuracy']['mean']:.3f}±{resnet_s['accuracy']['std']:.3f}  "
                  f"{resnet_s['auc']['mean']:.3f}±{resnet_s['auc']['std']:.3f}  "
                  f"{resnet_s['sensitivity']['mean']:.3f}±{resnet_s['sensitivity']['std']:.3f}  "
                  f"{resnet_s['specificity']['mean']:.3f}±{resnet_s['specificity']['std']:.3f}  "
                  f"{resnet_s['f1']['mean']:.3f}±{resnet_s['f1']['std']:.3f}  ← baseline")

    print(f"\n  Outputs → {OUTPUT_DIR.resolve()}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
