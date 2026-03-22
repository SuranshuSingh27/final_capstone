"""
STEP 5A — MSI HYPERPARAMETER TUNING (Optuna)
=============================================
Tunes hyperparameters for MSI deep models:
  - MSI_SpectralAttention
  - MSI_ResNet
  - MSI_3DCNN

Search space:
  - Learning rate
  - Dropout rate
  - Weight decay
  - Batch size
  - Class weight ratio

Strategy:
  - 20 Optuna trials per model (feasible on RTX 3050)
  - Each trial runs fold 1 only (fast proxy)
  - Best hyperparams retrained on all 5 folds
  - Results saved separately — does NOT overwrite step4 results

Usage:  python step5a_msi_tuning.py
Outputs → results/msi_tuned/
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

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from step2_preprocessing import OralMSIDataset

# ─────────────────────────────────────────────
SPLITS_DIR  = Path("results/splits")
OUTPUT_DIR  = Path("results/msi_tuned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS     = 5
MAX_EPOCHS  = 60       # per trial — fast
FULL_EPOCHS = 100      # final training with best params
PATIENCE    = 10       # aggressive for tuning
FULL_PATIENCE = 20     # generous for final training
N_TRIALS    = 20       # Optuna trials per model
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
# MODEL DEFINITIONS (parameterized for tuning)
# ══════════════════════════════════════════════
def build_spectral_attention(dropout=0.5, dropout2=0.3):
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
                nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout2),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            attn = self.band_attention(x.mean(dim=[2, 3]))
            x    = x * attn.unsqueeze(-1).unsqueeze(-1)
            return self.classifier(self.spatial_cnn(x))

        def get_attention_weights(self, x):
            return self.band_attention(x.mean(dim=[2, 3]))

    return SpectralAttentionNet()


def build_resnet_msi(dropout=0.5, dropout2=0.3):
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
        nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout2),
        nn.Linear(64, 2),
    )
    return model


def build_3d_cnn(dropout=0.5, dropout2=0.3):
    class MSI3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1),
                nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
                nn.Conv3d(32, 64, 3, padding=1),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((2, 4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*2*4*4, 256),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout2),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.classifier(self.features(x.unsqueeze(1)))
    return MSI3DCNN()


MODEL_BUILDERS = {
    "MSI_SpectralAttention": build_spectral_attention,
    "MSI_ResNet"           : build_resnet_msi,
    "MSI_3DCNN"            : build_3d_cnn,
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
    return total_loss / len(loader), np.array(all_labels), np.array(all_probs)


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
    return (total_loss / len(loader),
            np.array(all_labels), np.array(all_probs))


# ══════════════════════════════════════════════
# OPTUNA OBJECTIVE — runs on fold 1 only
# ══════════════════════════════════════════════
def make_objective(model_name, build_fn):
    def objective(trial):
        # Hyperparameter search space
        lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout", 0.2, 0.6)
        dropout2     = trial.suggest_float("dropout2", 0.1, 0.4)
        batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16])
        cw_cancer    = trial.suggest_float("cw_cancer", 1.5, 6.0)

        clear_cache()

        train_df = pd.read_csv(SPLITS_DIR / "fold1_train.csv")
        val_df   = pd.read_csv(SPLITS_DIR / "fold1_val.csv")

        train_ds = OralMSIDataset(train_df, training=True)
        val_ds   = OralMSIDataset(val_df,   training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

        model = build_fn(dropout=dropout, dropout2=dropout2).to(DEVICE)

        cw        = torch.tensor([0.5, cw_cancer], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw)

        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["fc","classifier","band_attention"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params,     "lr": lr},
        ], weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-7
        )

        best_auc     = -1
        patience_ctr = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, train_loader, criterion, optimizer)
            _, v_lab, v_prob = val_epoch(model, val_loader, criterion)
            scheduler.step()

            try:
                auc = roc_auc_score(v_lab, v_prob)
            except Exception:
                auc = 0.0

            if auc > best_auc:
                best_auc     = auc
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

            # Optuna pruning
            trial.report(best_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del model
        clear_cache()
        return best_auc

    return objective


# ══════════════════════════════════════════════
# TUNE ONE MODEL
# ══════════════════════════════════════════════
def tune_model(model_name, build_fn):
    print(f"\n{'='*55}")
    print(f"  Tuning: {model_name}  ({N_TRIALS} trials)")
    print(f"{'='*55}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        make_objective(model_name, build_fn),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n  Best trial: AUC={best.value:.4f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k:15s}: {v}")

    return best.params, best.value, study


# ══════════════════════════════════════════════
# RETRAIN WITH BEST PARAMS — ALL FOLDS
# ══════════════════════════════════════════════
def retrain_with_best(model_name, build_fn, best_params):
    print(f"\n{'='*55}")
    print(f"  Retraining: {model_name} with best params")
    print(f"{'='*55}")

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)

    lr           = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    dropout      = best_params["dropout"]
    dropout2     = best_params["dropout2"]
    batch_size   = best_params["batch_size"]
    cw_cancer    = best_params["cw_cancer"]

    cw        = torch.tensor([0.5, cw_cancer], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n  --- Fold {fold}/{N_FOLDS} ---")
        clear_cache()

        train_df = pd.read_csv(SPLITS_DIR / f"fold{fold}_train.csv")
        val_df   = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")

        train_ds = OralMSIDataset(train_df, training=True)
        val_ds   = OralMSIDataset(val_df,   training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

        model = build_fn(dropout=dropout, dropout2=dropout2).to(DEVICE)

        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ["fc","classifier","band_attention"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params,     "lr": lr},
        ], weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FULL_EPOCHS, eta_min=1e-7
        )

        best_auc         = -1
        best_model_state = None
        best_metrics     = None
        best_labels      = None
        best_probs       = None
        patience_counter = 0

        history = {
            "train_auc":[], "val_auc":[],
            "val_sens" :[], "val_acc":[],
        }

        for epoch in range(1, FULL_EPOCHS + 1):
            t0 = time.time()
            train_loss, tr_lab, tr_prob = train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, v_lab, v_prob = val_epoch(
                model, val_loader, criterion
            )
            scheduler.step()

            try:
                tr_auc = roc_auc_score(tr_lab, tr_prob)
                v_auc  = roc_auc_score(v_lab,  v_prob)
            except Exception:
                tr_auc = v_auc = 0.0

            v_m_05 = compute_metrics(v_lab, v_prob, 0.5)
            v_m_03 = compute_metrics(v_lab, v_prob, 0.3)

            history["train_auc"].append(tr_auc)
            history["val_auc"].append(v_auc)
            history["val_sens"].append(v_m_03["sensitivity"])
            history["val_acc"].append(v_m_05["accuracy"])

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{FULL_EPOCHS} | "
                  f"Loss {train_loss:.3f} | "
                  f"AUC {tr_auc:.3f}/{v_auc:.3f} | "
                  f"Acc {v_m_05['accuracy']:.3f} | "
                  f"Sens@0.3 {v_m_03['sensitivity']:.3f} | "
                  f"Spec@0.3 {v_m_03['specificity']:.3f} | "
                  f"{elapsed:.1f}s")

            if v_auc > best_auc:
                best_auc         = v_auc
                best_model_state = deepcopy(model.state_dict())
                best_metrics     = v_m_05
                best_labels      = v_lab.copy()
                best_probs       = v_prob.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= FULL_PATIENCE:
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
    print(f"  {model_name} — 5-Fold CV (tuned)")
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
    epochs = range(1, len(history["train_auc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} (tuned) — Fold {fold}",
                 fontsize=12, fontweight="bold")

    axes[0].plot(epochs, history["train_auc"], label="Train AUC", color="#534AB7")
    axes[0].plot(epochs, history["val_auc"],   label="Val AUC",   color="#D85A30")
    axes[0].axhline(0.85, color="green", linestyle="--", alpha=0.5)
    axes[0].set_title("AUC (early stopping metric)")
    axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"],  label="Val Acc",    color="#D85A30")
    axes[1].plot(epochs, history["val_sens"], label="Sens@0.3",   color="#1D9E75",
                 linestyle="--")
    axes[1].axhline(0.85, color="green", linestyle="--", alpha=0.5)
    axes[1].set_title("Accuracy + Sensitivity")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / f"fold{fold}_history.png",
                dpi=120, bbox_inches="tight")
    plt.close()


def plot_tuning_results(all_summaries, output_dir):
    """Compare tuned vs untuned MSI models."""
    # Load untuned results if available
    untuned_path = Path("results/msi_models/all_models_summary.json")
    has_untuned  = untuned_path.exists()

    metrics = ["accuracy","auc","sensitivity","specificity","f1"]
    colors  = ["#534AB7","#D85A30","#1D9E75"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("MSI Models — Tuned vs Untuned (5-fold CV)",
                 fontsize=12, fontweight="bold")

    if has_untuned:
        with open(untuned_path) as f:
            untuned = json.load(f)

    for ax, metric in zip(axes, metrics):
        model_names = list(all_summaries.keys())
        tuned_means = [all_summaries[mn][metric]["mean"] for mn in model_names]
        tuned_stds  = [all_summaries[mn][metric]["std"]  for mn in model_names]

        x     = np.arange(len(model_names))
        width = 0.35

        ax.bar(x - width/2 if has_untuned else x,
               tuned_means, width if has_untuned else 0.6,
               label="Tuned", color="#534AB7", alpha=0.85,
               yerr=tuned_stds, capsize=3)

        if has_untuned:
            untuned_means = []
            untuned_stds  = []
            for mn in model_names:
                if mn in untuned:
                    untuned_means.append(untuned[mn][metric]["mean"])
                    untuned_stds.append(untuned[mn][metric]["std"])
                else:
                    untuned_means.append(0)
                    untuned_stds.append(0)
            ax.bar(x + width/2, untuned_means, width,
                   label="Untuned", color="#D85A30", alpha=0.85,
                   yerr=untuned_stds, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([mn.replace("MSI_","") for mn in model_names],
                           rotation=15, fontsize=8)
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.1)
        ax.axhline(0.85, color="green", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == metrics[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "tuned_vs_untuned_msi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot → {out}")


def plot_optuna_history(study, model_name, output_dir):
    """Plot Optuna optimization history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Optuna — {model_name}", fontsize=12, fontweight="bold")

    # Optimization history
    values = [t.value for t in study.trials if t.value is not None]
    axes[0].plot(values, color="#534AB7", marker="o", markersize=4)
    axes[0].axhline(max(values), color="red", linestyle="--", alpha=0.5,
                    label=f"Best={max(values):.3f}")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("AUC (fold 1)")
    axes[0].set_title("Optimization history")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Parameter importance (top 5)
    try:
        importance = optuna.importance.get_param_importances(study)
        params     = list(importance.keys())[:5]
        values_imp = [importance[p] for p in params]
        axes[1].barh(params, values_imp, color="#534AB7", alpha=0.85)
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Hyperparameter importance")
        axes[1].grid(True, alpha=0.3, axis="x")
    except Exception:
        axes[1].text(0.5, 0.5, "Not enough trials\nfor importance",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    out = output_dir / model_name / "optuna_history.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curves(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = {
        "MSI_SpectralAttention": "#D85A30",
        "MSI_ResNet"           : "#534AB7",
        "MSI_3DCNN"            : "#3B8BD4",
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
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — MSI Tuned Models",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "roc_curves_msi_tuned.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ROC curves → {out}")


def plot_confusion_matrices(all_results, output_dir):
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Confusion Matrices — MSI Tuned Models",
                 fontsize=12, fontweight="bold")
    for col, (mn, (all_labels, all_probs)) in enumerate(all_results.items()):
        for row, t in enumerate([0.5, 0.3]):
            ax    = axes[row][col]
            preds = (all_probs >= t).astype(int)
            cm    = confusion_matrix(all_labels, preds, labels=[0,1])
            ax.imshow(cm, cmap="Purples")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Non-C","Cancer"], fontsize=9)
            ax.set_yticklabels(["Non-C","Cancer"], fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{mn.replace('MSI_','')} (t={t})", fontsize=9)
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i,j] > cm.max()/2 else "black"
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    out = output_dir / "confusion_matrices_msi_tuned.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrices → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  STEP 5A — MSI HYPERPARAMETER TUNING")
    print("█"*55)
    print(f"  Trials per model : {N_TRIALS}")
    print(f"  Tuning epochs    : {MAX_EPOCHS}")
    print(f"  Final epochs     : {FULL_EPOCHS}")
    print(f"  Device           : {DEVICE}")
    print(f"  Output           : {OUTPUT_DIR}")
    print(f"  NOTE: Does NOT overwrite results/msi_models/")

    all_best_params = {}
    all_summaries   = {}
    all_results     = {}

    for model_name, build_fn in MODEL_BUILDERS.items():
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(exist_ok=True)

        # Tune
        best_params, best_val_auc, study = tune_model(model_name, build_fn)
        all_best_params[model_name] = {
            "params"      : best_params,
            "best_val_auc": best_val_auc,
        }

        # Save best params
        with open(model_dir / "best_params.json", "w") as f:
            json.dump({"params": best_params, "best_val_auc": best_val_auc},
                      f, indent=2)

        # Plot Optuna history
        plot_optuna_history(study, model_name, OUTPUT_DIR)

        # Retrain with best params
        fold_results = retrain_with_best(model_name, build_fn, best_params)
        summary, all_labels, all_probs = aggregate_results(
            fold_results, model_name
        )
        all_summaries[model_name] = summary
        all_results[model_name]   = (all_labels, all_probs)

        with open(model_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Save all summaries
    with open(OUTPUT_DIR / "all_models_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    with open(OUTPUT_DIR / "all_best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=2)

    # Plots
    print("\n  Generating plots...")
    plot_roc_curves(all_results, OUTPUT_DIR)
    plot_confusion_matrices(all_results, OUTPUT_DIR)
    plot_tuning_results(all_summaries, OUTPUT_DIR)

    # Final summary
    print("\n" + "="*65)
    print("  TUNED MSI RESULTS SUMMARY")
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

    print(f"\n  Best params saved → {OUTPUT_DIR / 'all_best_params.json'}")
    print(f"  Outputs → {OUTPUT_DIR.resolve()}")
    print("  Next    → python step5b_fusion_model.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
