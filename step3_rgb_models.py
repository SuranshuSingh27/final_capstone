"""
STEP 3 — RGB MODELS (stable version)
======================================
Strategy: Get the best accuracy first, then analyze
sensitivity/specificity via threshold tuning in post-processing.

This is the approach that gave ResNet 84.9% and EfficientNet 83.2%.

Models:
  1. Custom CNN        (baseline)
  2. ResNet-50         (transfer learning)
  3. EfficientNet-B0   (transfer learning)

Training:
  - Standard CrossEntropyLoss with balanced class weights
  - Early stopping on val ACCURACY
  - AdamW + CosineAnnealingLR
  - Differential LR: backbone=LR/10, head=LR
  - After training: analyze metrics at BOTH 0.5 and 0.3 thresholds

Usage:  python step3_rgb_models.py
Outputs → results/rgb_models/
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
from step2_preprocessing import OralRGBDataset

# ─────────────────────────────────────────────
SPLITS_DIR = Path("results/splits")
OUTPUT_DIR = Path("results/rgb_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS    = 5
EPOCHS     = 100
BATCH_SIZE = 16
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
# METRICS at multiple thresholds
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
    """Compute metrics at multiple thresholds for post-hoc analysis."""
    results = {}
    for t in [0.3, 0.4, 0.5]:
        results[str(t)] = compute_metrics(labels, probs, t)
    return results


# ══════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════
def build_custom_cnn():
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.1),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.15),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(0.2),

                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
        def forward(self, x):
            return self.classifier(self.features(x))
    return CustomCNN()


def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for layer in [model.layer3, model.layer4]:
        for param in layer.parameters():
            param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return model


def build_efficientnet():
    model = EfficientNet.from_pretrained("efficientnet-b0")
    for param in model.parameters():
        param.requires_grad = False
    for block in model._blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model._conv_head.parameters():
        param.requires_grad = True
    for param in model._bn1.parameters():
        param.requires_grad = True
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    return model


MODELS = {
    "CustomCNN"   : build_custom_cnn,
    "ResNet50"    : build_resnet50,
    "EfficientNet": build_efficientnet,
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
    metrics    = compute_metrics(labels_arr, probs_arr, threshold=0.5)
    return total_loss / len(loader), metrics, labels_arr, probs_arr


# ══════════════════════════════════════════════
# TRAIN ONE MODEL — ALL FOLDS
# ══════════════════════════════════════════════
def train_model(model_name, build_fn, class_weights):
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

        train_ds = OralRGBDataset(train_df, training=True)
        val_ds   = OralRGBDataset(val_df,   training=False)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=False
        )

        model = build_fn().to(DEVICE)

        # Differential LR
        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["fc", "_fc", "classifier"]):
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
            "train_loss": [], "val_loss": [],
            "train_acc" : [], "val_acc" : [],
            "train_auc" : [], "val_auc" : [],
            "train_sens": [], "val_sens": [],
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
            history["train_sens"].append(train_m["sensitivity"])
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

        # Threshold analysis
        thresh_analysis = compute_threshold_analysis(best_labels, best_probs)

        print(f"\n  Fold {fold} best results:")
        for t, m in thresh_analysis.items():
            print(f"    Threshold={t}: "
                  f"Acc={m['accuracy']:.3f} "
                  f"Sens={m['sensitivity']:.3f} "
                  f"Spec={m['specificity']:.3f} "
                  f"AUC={m['auc']:.3f} "
                  f"F1={m['f1']:.3f} "
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

    keys = ["accuracy", "auc", "sensitivity", "specificity", "f1"]
    agg  = {k: [] for k in keys}
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

    # Threshold analysis summary
    print(f"\n  Threshold sensitivity analysis:")
    for t in ["0.3", "0.4", "0.5"]:
        sens_vals = [fr["thresh_analysis"][t]["sensitivity"]
                     for fr in fold_results]
        spec_vals = [fr["thresh_analysis"][t]["specificity"]
                     for fr in fold_results]
        acc_vals  = [fr["thresh_analysis"][t]["accuracy"]
                     for fr in fold_results]
        print(f"    t={t}: "
              f"Acc={np.mean(acc_vals):.3f} "
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
# PLOTS
# ══════════════════════════════════════════════
def plot_training_history(history, fold, model_name, model_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Fold {fold}", fontsize=12, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#3B8BD4")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#D85A30")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#3B8BD4")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#D85A30")
    axes[1].axhline(0.85, color="green",  linestyle="--", alpha=0.5, label="85%")
    axes[1].axhline(0.90, color="orange", linestyle="--", alpha=0.5, label="90%")
    axes[1].set_title("Accuracy (early stopping metric)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_auc"], label="Train AUC", color="#3B8BD4")
    axes[2].plot(epochs, history["val_auc"],   label="Val AUC",   color="#D85A30")
    axes[2].plot(epochs, history["val_sens"],  label="Val Sens",
                 color="#1D9E75", linestyle="--")
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
        "CustomCNN"   : "#534AB7",
        "ResNet50"    : "#3B8BD4",
        "EfficientNet": "#D85A30",
    }
    for model_name, (all_labels, all_probs) in all_results.items():
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc         = roc_auc_score(all_labels, all_probs)
        ax.plot(fpr, tpr, color=colors.get(model_name, "#888"),
                linewidth=2, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — RGB Models (5-fold pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "roc_curves_rgb.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curves → {out}")


def plot_confusion_matrices(all_results, output_dir):
    # Show at both 0.5 and 0.3 thresholds
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Confusion Matrices — RGB Models",
                 fontsize=12, fontweight="bold")

    for col, (model_name, (all_labels, all_probs)) in enumerate(all_results.items()):
        for row, t in enumerate([0.5, 0.3]):
            ax    = axes[row][col]
            preds = (all_probs >= t).astype(int)
            cm    = confusion_matrix(all_labels, preds, labels=[0,1])
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Non-Cancer","Cancer"])
            ax.set_yticklabels(["Non-Cancer","Cancer"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{model_name} (t={t})")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i,j] > cm.max()/2 else "black"
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            fontsize=13, fontweight="bold", color=color)

    plt.tight_layout()
    out = output_dir / "confusion_matrices_rgb.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrices → {out}")


def plot_model_comparison(all_summaries, output_dir):
    metrics = ["accuracy","auc","sensitivity","specificity","f1"]
    x, width = np.arange(len(metrics)), 0.25
    colors   = ["#534AB7","#3B8BD4","#D85A30"]
    fig, ax  = plt.subplots(figsize=(13, 6))
    for i, (mn, color) in enumerate(zip(all_summaries.keys(), colors)):
        means = [all_summaries[mn][m]["mean"] for m in metrics]
        stds  = [all_summaries[mn][m]["std"]  for m in metrics]
        ax.bar(x+i*width, means, width, label=mn, color=color,
               alpha=0.85, yerr=stds, capsize=4)
    ax.set_xticks(x+width)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RGB Models — 5-fold CV Comparison (threshold=0.5)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.85, color="green",  linestyle="--", alpha=0.4, label="85%")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.4, label="90%")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / "model_comparison_rgb.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Comparison → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  STEP 3 — RGB MODELS (stable)")
    print("█"*55)
    print(f"  Strategy   : Maximize accuracy first")
    print(f"  Threshold  : 0.5 for training + stopping")
    print(f"  Post-hoc   : Also reports at 0.3 and 0.4")
    print(f"  Epochs     : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : backbone={LR*0.1:.0e}  head={LR:.0e}")
    print(f"  Device     : {DEVICE}")

    # Load class weights from step 2 (balanced, not aggressive)
    class_weights = torch.load(
        SPLITS_DIR / "class_weights.pt", weights_only=True
    )
    print(f"  Class weights: {[round(w,4) for w in class_weights.tolist()]}")

    all_summaries = {}
    all_results   = {}

    for model_name, build_fn in MODELS.items():
        fold_results = train_model(model_name, build_fn, class_weights)
        summary, all_labels, all_probs = aggregate_results(
            fold_results, model_name
        )
        all_summaries[model_name] = summary
        all_results[model_name]   = (all_labels, all_probs)

        with open(OUTPUT_DIR / model_name / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n  Generating plots...")
    plot_roc_curves(all_results, OUTPUT_DIR)
    plot_confusion_matrices(all_results, OUTPUT_DIR)
    plot_model_comparison(all_summaries, OUTPUT_DIR)

    print("\n" + "="*65)
    print("  FINAL RGB RESULTS SUMMARY (threshold=0.5)")
    print("="*65)
    print(f"  {'Model':15s} {'Accuracy':>10} {'AUC':>8} "
          f"{'Sens':>8} {'Spec':>8} {'F1':>8}")
    print("  " + "─"*55)
    for mn, s in all_summaries.items():
        print(f"  {mn:15s} "
              f"{s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}  "
              f"{s['auc']['mean']:.3f}±{s['auc']['std']:.3f}  "
              f"{s['sensitivity']['mean']:.3f}±{s['sensitivity']['std']:.3f}  "
              f"{s['specificity']['mean']:.3f}±{s['specificity']['std']:.3f}  "
              f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}")

    with open(OUTPUT_DIR / "all_models_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n  Outputs → {OUTPUT_DIR.resolve()}")
    print("  Next    → python step4_msi_models.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
