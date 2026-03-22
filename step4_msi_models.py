"""
STEP 4 — MSI MODELS (stable version)
=======================================
Same stable strategy as step3:
  - Standard class weights from step2
  - Early stopping on accuracy
  - No weighted sampler, no aggressive oversampling
  - Post-hoc threshold analysis at 0.3, 0.4, 0.5
  - SMOTE only for classical ML (16-D vectors — safe)

Models:
  1. SVM                   — 16-D spectral vectors + SMOTE
  2. Random Forest          — 16-D spectral vectors + SMOTE
  3. 3D CNN                — full spectral cube
  4. CNN-LSTM              — bands as sequence
  5. Spectral Attention     — learns band importance (key contribution)
  6. ResNet-MSI            — ResNet for 16 channels

Usage:  python step4_msi_models.py
Outputs → results/msi_models/
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
from torch.utils.data import DataLoader
import torchvision.models as models

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    f1_score, roc_curve
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("ERROR: pip install imbalanced-learn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from step2_preprocessing import OralMSIDataset, OralSpectralVectorDataset

# ─────────────────────────────────────────────
SPLITS_DIR = Path("results/splits")
OUTPUT_DIR = Path("results/msi_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS    = 5
EPOCHS     = 100
BATCH_SIZE = 8
LR         = 1e-4
PATIENCE   = 20
SEED       = 42

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
    results = {}
    for t in [0.3, 0.4, 0.5]:
        results[str(t)] = compute_metrics(labels, probs, t)
    return results


# ══════════════════════════════════════════════
# CLASSICAL ML — SMOTE on 16-D (safe)
# ══════════════════════════════════════════════
def extract_vectors(df):
    ds      = OralSpectralVectorDataset(df)
    vectors = []
    labels  = []
    for i in range(len(ds)):
        s = ds[i]
        vectors.append(s["vector"].numpy())
        labels.append(s["label"].item())
    return np.array(vectors), np.array(labels)


def train_classical(model_name, clf):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name} (classical ML + SMOTE on 16-D)")
    print(f"{'='*55}")

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)

    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n  --- Fold {fold}/{N_FOLDS} ---")

        train_df = pd.read_csv(SPLITS_DIR / f"fold{fold}_train.csv")
        val_df   = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")

        X_train, y_train = extract_vectors(train_df)
        X_val,   y_val   = extract_vectors(val_df)

        n_cancer = (y_train == 1).sum()
        print(f"  Before SMOTE: cancer={n_cancer} "
              f"non-cancer={(y_train==0).sum()}")

        smote    = SMOTE(
            sampling_strategy=0.7,
            random_state=SEED,
            k_neighbors=min(5, n_cancer - 1)
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"  After  SMOTE: cancer={(y_res==1).sum()} "
              f"non-cancer={(y_res==0).sum()}")

        clf.fit(X_res, y_res)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_val)[:, 1]
        else:
            scores = clf.decision_function(X_val)
            probs  = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        thresh_analysis = compute_threshold_analysis(y_val, probs)

        print(f"  Fold {fold} results:")
        for t, m in thresh_analysis.items():
            print(f"    t={t}: Acc={m['accuracy']:.3f} "
                  f"Sens={m['sensitivity']:.3f} "
                  f"Spec={m['specificity']:.3f} "
                  f"AUC={m['auc']:.3f} "
                  f"TP={m['tp']} FN={m['fn']}")

        fold_results.append({
            "fold"           : fold,
            "metrics"        : thresh_analysis["0.5"],
            "thresh_analysis": thresh_analysis,
            "labels"         : y_val,
            "probs"          : probs,
        })

    return fold_results


# ══════════════════════════════════════════════
# DEEP MODEL DEFINITIONS
# ══════════════════════════════════════════════
def build_3d_cnn():
    class MSI3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1),
                nn.BatchNorm3d(16), nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, 3, padding=1),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((2, 4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*2*4*4, 256),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.classifier(self.features(x.unsqueeze(1)))
    return MSI3DCNN()


def build_cnn_lstm():
    class MSILSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.band_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32*4*4, 64), nn.ReLU(inplace=True),
            )
            self.lstm = nn.LSTM(64, 128, num_layers=2,
                                batch_first=True, dropout=0.3,
                                bidirectional=True)
            self.classifier = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.4), nn.Linear(64, 2),
            )
        def forward(self, x):
            B, C, H, W = x.shape
            feats = torch.stack(
                [self.band_cnn(x[:, i:i+1]) for i in range(C)], dim=1
            )
            out, _ = self.lstm(feats)
            return self.classifier(out[:, -1])
    return MSILSTM()


def build_spectral_attention():
    class SpectralAttentionNet(nn.Module):
        def __init__(self, n_bands=16):
            super().__init__()
            self.band_attention = nn.Sequential(
                nn.Linear(n_bands, 32), nn.ReLU(inplace=True),
                nn.Linear(32, n_bands), nn.Softmax(dim=-1),
            )
            self.spatial_cnn = nn.Sequential(
                nn.Conv2d(n_bands, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*4*4, 256),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            attn = self.band_attention(x.mean(dim=[2, 3]))
            x    = x * attn.unsqueeze(-1).unsqueeze(-1)
            return self.classifier(self.spatial_cnn(x))
        def get_attention_weights(self, x):
            return self.band_attention(x.mean(dim=[2, 3]))
    return SpectralAttentionNet()


def build_resnet_msi():
    model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv = model.conv1
    new_conv = nn.Conv2d(16, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding, bias=False)
    with torch.no_grad():
        avg_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight = nn.Parameter(avg_w.repeat(1, 16, 1, 1))
    model.conv1 = new_conv
    for name, param in model.named_parameters():
        if "layer1" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(64, 2),
    )
    return model


DEEP_MODELS = {
    "MSI_3DCNN"           : build_3d_cnn,
    "MSI_CNNLSTM"         : build_cnn_lstm,
    "MSI_SpectralAttention": build_spectral_attention,
    "MSI_ResNet"          : build_resnet_msi,
}


# ══════════════════════════════════════════════
# TRAIN / VAL EPOCH
# ══════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
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
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs  = model(images)
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
# TRAIN ONE DEEP MODEL
# ══════════════════════════════════════════════
def train_deep_model(model_name, build_fn, class_weights):
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

        train_ds = OralMSIDataset(train_df, training=True)
        val_ds   = OralMSIDataset(val_df,   training=False)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=False
        )

        model = build_fn().to(DEVICE)

        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["fc","classifier","band_attention"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": LR * 0.1},
            {"params": head_params,     "lr": LR},
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

            # Early stopping on ACCURACY
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
        print(f"\n  Fold {fold} results:")
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
    print(f"  {model_name} — 5-Fold CV Summary")
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

    print(f"\n  Threshold sensitivity analysis:")
    for t in ["0.3","0.4","0.5"]:
        sens_vals = [fr["thresh_analysis"][t]["sensitivity"]
                     for fr in fold_results]
        spec_vals = [fr["thresh_analysis"][t]["specificity"]
                     for fr in fold_results]
        acc_vals  = [fr["thresh_analysis"][t]["accuracy"]
                     for fr in fold_results]
        print(f"    t={t}: Acc={np.mean(acc_vals):.3f} "
              f"Sens={np.mean(sens_vals):.3f}±{np.std(sens_vals):.3f} "
              f"Spec={np.mean(spec_vals):.3f}±{np.std(spec_vals):.3f}")

    all_labels = np.concatenate([fr["labels"] for fr in fold_results])
    all_probs  = np.concatenate([fr["probs"]  for fr in fold_results])
    try:
        print(f"\n  Pooled AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    except Exception:
        pass

    return summary, all_labels, all_probs


# ══════════════════════════════════════════════
# BAND IMPORTANCE
# ══════════════════════════════════════════════
def analyze_band_importance():
    print(f"\n{'='*55}")
    print(f"  Band importance — MSI_SpectralAttention")
    print(f"{'='*55}")

    model_dir = OUTPUT_DIR / "MSI_SpectralAttention"
    if not model_dir.exists():
        print("  Skipping — not trained yet")
        return

    wavelengths = [460,465,474,483,493,504,512,522,
                   534,541,552,560,570,580,585,595]
    all_attn = []

    for fold in range(1, N_FOLDS + 1):
        wp = model_dir / f"fold{fold}_best.pt"
        if not wp.exists():
            continue
        model = build_spectral_attention().to(DEVICE)
        model.load_state_dict(
            torch.load(wp, map_location=DEVICE, weights_only=True)
        )
        model.eval()

        val_df = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")
        val_ds = OralMSIDataset(val_df, training=False)
        loader = DataLoader(val_ds, batch_size=4,
                            shuffle=False, num_workers=0)

        fold_attn = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(DEVICE)
                attn = model.get_attention_weights(imgs)
                fold_attn.append(attn.cpu().numpy())
        all_attn.append(np.concatenate(fold_attn).mean(axis=0))

    if not all_attn:
        return

    mean_attn = np.array(all_attn).mean(axis=0)
    std_attn  = np.array(all_attn).std(axis=0)

    print(f"\n  Wavelength importance:")
    for wl, a, s in zip(wavelengths, mean_attn, std_attn):
        bar = "█" * int(a * 200)
        print(f"  {wl}nm : {a:.4f} ± {s:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(16), mean_attn, yerr=std_attn,
           color="#534AB7", alpha=0.8, capsize=4)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{w}nm" for w in wavelengths],
                       rotation=45, fontsize=9)
    ax.set_ylabel("Attention weight", fontsize=11)
    ax.set_title("Spectral Band Importance for OSCC Detection",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = OUTPUT_DIR / "band_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")

    with open(OUTPUT_DIR / "band_importance.json", "w") as f:
        json.dump({
            "wavelengths"   : wavelengths,
            "mean_attention": mean_attn.tolist(),
            "std_attention" : std_attn.tolist(),
        }, f, indent=2)


# ══════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════
def plot_training_history(history, fold, model_name, model_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Fold {fold}",
                 fontsize=12, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#534AB7")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#D85A30")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#534AB7")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#D85A30")
    axes[1].axhline(0.85, color="green",  linestyle="--", alpha=0.5, label="85%")
    axes[1].axhline(0.90, color="orange", linestyle="--", alpha=0.5, label="90%")
    axes[1].set_title("Accuracy (early stopping)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_auc"], label="Train AUC", color="#534AB7")
    axes[2].plot(epochs, history["val_auc"],   label="Val AUC",   color="#D85A30")
    axes[2].plot(epochs, history["val_sens"],
                 label="Val Sens", color="#1D9E75", linestyle="--")
    axes[2].axhline(0.85, color="green", linestyle="--", alpha=0.5)
    axes[2].set_title("AUC + Sensitivity")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / f"fold{fold}_history.png",
                dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curves(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = {
        "SVM":"#1D9E75","RandomForest":"#888780",
        "MSI_3DCNN":"#534AB7","MSI_CNNLSTM":"#3B8BD4",
        "MSI_SpectralAttention":"#D85A30","MSI_ResNet":"#BA7517",
    }
    for mn, (all_labels, all_probs) in all_results.items():
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            auc         = roc_auc_score(all_labels, all_probs)
            ax.plot(fpr, tpr, color=colors.get(mn,"#888"),
                    linewidth=2, label=f"{mn} (AUC={auc:.3f})")
        except Exception:
            continue
    ax.plot([0,1],[0,1],"k--",alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — MSI Models (5-fold pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "roc_curves_msi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curves → {out}")


def plot_confusion_matrices(all_results, output_dir):
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Confusion Matrices — MSI Models",
                 fontsize=12, fontweight="bold")

    for col, (mn, (all_labels, all_probs)) in enumerate(all_results.items()):
        for row, t in enumerate([0.5, 0.3]):
            ax    = axes[row][col]
            preds = (all_probs >= t).astype(int)
            cm    = confusion_matrix(all_labels, preds, labels=[0,1])
            ax.imshow(cm, cmap="Purples")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Non-C","Cancer"], fontsize=8)
            ax.set_yticklabels(["Non-C","Cancer"], fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{mn} (t={t})", fontsize=8)
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i,j] > cm.max()/2 else "black"
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    out = output_dir / "confusion_matrices_msi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrices → {out}")


def plot_rgb_vs_msi(msi_summaries, output_dir):
    rgb_path = Path("results/rgb_models/all_models_summary.json")
    if not rgb_path.exists():
        print("  RGB summary not found — skipping")
        return
    with open(rgb_path) as f:
        rgb_summaries = json.load(f)

    metrics      = ["accuracy","auc","sensitivity","specificity","f1"]
    best_rgb_name= max(rgb_summaries,
                       key=lambda k: rgb_summaries[k]["auc"]["mean"])
    best_msi_name= max(msi_summaries,
                       key=lambda k: msi_summaries[k]["auc"]["mean"])
    best_rgb     = rgb_summaries[best_rgb_name]
    best_msi     = msi_summaries[best_msi_name]

    x, width = np.arange(len(metrics)), 0.35
    fig, ax  = plt.subplots(figsize=(12, 6))
    ax.bar(x-width/2, [best_rgb[m]["mean"] for m in metrics], width,
           label=f"Best RGB ({best_rgb_name})", color="#3B8BD4", alpha=0.85,
           yerr=[best_rgb[m]["std"] for m in metrics], capsize=4)
    ax.bar(x+width/2, [best_msi[m]["mean"] for m in metrics], width,
           label=f"Best MSI ({best_msi_name})", color="#534AB7", alpha=0.85,
           yerr=[best_msi[m]["std"] for m in metrics], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RGB vs MSI — Best Models Comparison (5-fold CV)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.85, color="green",  linestyle="--", alpha=0.4, label="85%")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.4, label="90%")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / "rgb_vs_msi_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  RGB vs MSI → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  STEP 4 — MSI MODELS (stable)")
    print("█"*55)
    print(f"  Strategy  : Accuracy-first, post-hoc threshold analysis")
    print(f"  Classical : 16-D vectors + SMOTE")
    print(f"  Deep      : Standard training, no oversampling")
    print(f"  Epochs    : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Device    : {DEVICE}")

    class_weights = torch.load(
        SPLITS_DIR / "class_weights.pt", weights_only=True
    )
    print(f"  Class weights: {[round(w,4) for w in class_weights.tolist()]}")

    all_summaries = {}
    all_results   = {}

    # Classical ML
    print("\n  CLASSICAL ML")
    classical = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced",
                        probability=True, random_state=SEED)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1)),
        ]),
    }
    for mn, clf in classical.items():
        fold_results = train_classical(mn, clf)
        summary, all_labels, all_probs = aggregate_results(fold_results, mn)
        all_summaries[mn] = summary
        all_results[mn]   = (all_labels, all_probs)
        model_dir = OUTPUT_DIR / mn
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Deep Learning
    print("\n  DEEP LEARNING MSI")
    for mn, build_fn in DEEP_MODELS.items():
        fold_results = train_deep_model(mn, build_fn, class_weights)
        summary, all_labels, all_probs = aggregate_results(fold_results, mn)
        all_summaries[mn] = summary
        all_results[mn]   = (all_labels, all_probs)
        with open(OUTPUT_DIR / mn / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    analyze_band_importance()

    print("\n  Generating plots...")
    plot_roc_curves(all_results, OUTPUT_DIR)
    plot_confusion_matrices(all_results, OUTPUT_DIR)
    plot_rgb_vs_msi(all_summaries, OUTPUT_DIR)

    print("\n" + "="*65)
    print("  FINAL MSI RESULTS (threshold=0.5)")
    print("="*65)
    print(f"  {'Model':25s} {'Acc':>8} {'AUC':>8} "
          f"{'Sens':>8} {'Spec':>8} {'F1':>8}")
    print("  " + "─"*62)
    for mn, s in all_summaries.items():
        print(f"  {mn:25s} "
              f"{s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}  "
              f"{s['auc']['mean']:.3f}±{s['auc']['std']:.3f}  "
              f"{s['sensitivity']['mean']:.3f}±{s['sensitivity']['std']:.3f}  "
              f"{s['specificity']['mean']:.3f}±{s['specificity']['std']:.3f}  "
              f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}")

    with open(OUTPUT_DIR / "all_models_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n  Outputs → {OUTPUT_DIR.resolve()}")
    print("  Next    → step5_analysis.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
