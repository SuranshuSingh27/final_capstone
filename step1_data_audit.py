"""
STEP 1 — DATA AUDIT (final fixed version)
==========================================
Usage:  python3 step1_data_audit.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

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
OUTPUT_DIR    = Path("results/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────

# All diagnosis values found in your Excel (stripped + lowercased)
LABEL_MAP = {
    "healthy"    : 0,
    "control"    : 0,
    "smoker"     : 0,
    "osmf"       : 0,
    "leukoplakia": 0,
    "keratosis"  : 0,   
    "oscc"       : 1,
}
CLASS_NAMES  = {0: "Non-Cancer", 1: "Cancer (OSCC)"}
CLASS_COLORS = {0: "#3B8BD4",    1: "#D85A30"}


def safe_label(x):
    """Map diagnosis string to binary label. Handles NaN and trailing spaces."""
    if pd.isna(x):
        return -1
    cleaned = str(x).strip().lower()
    for k, v in LABEL_MAP.items():
        if k in cleaned:
            return v
    return -1


# ══════════════════════════════════════════════
# 1. LOAD EXCEL
# ══════════════════════════════════════════════
def load_labels():
    print("\n" + "="*55)
    print("  1/5  Reading labels from Excel")
    print("="*55)

    xl = pd.ExcelFile(EXCEL_PATH)

    # Sheet 1: image_id, patient_id, image_number
    df_img = xl.parse("Image ID", header=0)
    df_img.columns = df_img.columns.str.strip().str.lower().str.replace(" ", "_")
    df_img["image_id"]   = df_img["image_id"].astype(str).str.strip()
    df_img["patient_id"] = df_img["patient_id"].astype(str).str.strip()
    print(f"  Image ID rows   : {len(df_img)}")
    print(f"  Sample image_ids: {df_img['image_id'].head(5).tolist()}")

    # Sheet 2: real header is on row 1 (index 1), not row 0
    df_pat = xl.parse("Patient data", header=1)
    df_pat.columns = [
        str(c).strip().lower().replace(" ", "_") if not str(c).startswith("Unnamed") else f"_drop_{i}"
        for i, c in enumerate(df_pat.columns)
    ]
    # Keep only named columns
    df_pat = df_pat[[c for c in df_pat.columns if not c.startswith("_drop")]]
    df_pat = df_pat.dropna(subset=["patient_id"])
    df_pat["patient_id"] = df_pat["patient_id"].astype(str).str.strip()
    # Strip diagnosis
    df_pat["diagnosis"]  = df_pat["diagnosis"].astype(str).str.strip()

    print(f"\n  Patient rows    : {len(df_pat)}")
    print(f"  Diagnosis values: {sorted(df_pat['diagnosis'].dropna().unique())}")

    # Binary label — safe against NaN and trailing spaces
    df_pat["binary_label"] = df_pat["diagnosis"].map(safe_label)

    unmapped = df_pat[df_pat["binary_label"] == -1]["diagnosis"].unique()
    if len(unmapped):
        print(f"\n  WARNING unmapped diagnosis values: {unmapped}")
        print("  Add them to LABEL_MAP if needed.")

    # Merge
    df = df_img.merge(
        df_pat[["patient_id", "diagnosis", "binary_label",
                "gender", "age_range_(yrs)", "smoking"]],
        on="patient_id", how="left"
    )
    df["diag_raw"] = df["diagnosis"]

    print(f"\n  Merged rows     : {len(df)}")
    print(f"  Null labels     : {df['binary_label'].isna().sum()}")
    print(f"\n  Diagnosis counts:")
    print(df["diagnosis"].value_counts().to_string())

    return df


# ══════════════════════════════════════════════
# 2. BUILD FILE MANIFEST
# ══════════════════════════════════════════════
def build_manifest(df):
    print("\n" + "="*55)
    print("  2/5  Scanning files on disk")
    print("="*55)

    hdr_files  = sorted(PROCESSED_DIR.glob("**/*.hdr")) if PROCESSED_DIR.exists() else []
    rgb_files  = sorted(RGB_DIR.glob("**/*.*"))          if RGB_DIR.exists()       else []
    mask_files = sorted(MASK_DIR.glob("**/*.png"))       if MASK_DIR.exists()      else []

    print(f"  HDR  files : {len(hdr_files)}")
    print(f"  RGB  files : {len(rgb_files)}")
    print(f"  Mask files : {len(mask_files)}")

    # Show a few filenames so we know naming convention
    if hdr_files:
        print(f"\n  Sample HDR  names : {[f.stem for f in hdr_files[:5]]}")
    if rgb_files:
        print(f"  Sample RGB  names : {[f.stem for f in rgb_files[:5]]}")
    if mask_files:
        print(f"  Sample mask names : {[f.stem for f in mask_files[:5]]}")

    hdr_dict  = {f.stem: f for f in hdr_files}
    rgb_dict  = {f.stem: f for f in rgb_files}
    mask_dict = {f.stem: f for f in mask_files}

    records = []
    for _, row in df.iterrows():
        iid = str(row["image_id"]).strip()

        # Try multiple matching strategies:
        # 1. exact stem match
        # 2. stem contains image_id
        # 3. image_id contains stem
        def find_file(d):
            if iid in d:
                return d[iid]
            for k, v in d.items():
                if iid in k or k in iid:
                    return v
            return None

        hdr  = find_file(hdr_dict)
        rgb  = find_file(rgb_dict)
        mask = find_file(mask_dict)

        records.append({
            "image_id"    : iid,
            "patient_id"  : str(row.get("patient_id", "")),
            "binary_label": row.get("binary_label", -1),
            "diag_raw"    : row.get("diag_raw", ""),
            "gender"      : row.get("gender", ""),
            "age_range"   : row.get("age_range_(yrs)", ""),
            "smoking"     : row.get("smoking", ""),
            "hdr_path"    : hdr,
            "rgb_path"    : rgb,
            "mask_path"   : mask,
            "has_hdr"     : hdr  is not None,
            "has_rgb"     : rgb  is not None,
            "has_mask"    : mask is not None,
        })

    manifest = pd.DataFrame(records)
    complete = (manifest["has_hdr"] & manifest["has_rgb"] & manifest["has_mask"]).sum()

    print(f"\n  Complete triplets : {complete} / {len(manifest)}")
    print(f"  Missing HDR       : {(~manifest['has_hdr']).sum()}")
    print(f"  Missing RGB       : {(~manifest['has_rgb']).sum()}")
    print(f"  Missing mask      : {(~manifest['has_mask']).sum()}")

    # If nothing matched, diagnose the issue
    if complete == 0:
        print("\n  ⚠  No files matched — diagnosing naming mismatch...")
        print(f"  Excel image_ids (first 5) : {manifest['image_id'].head(5).tolist()}")
        print(f"  HDR stems      (first 5)  : {list(hdr_dict.keys())[:5]}")
        print(f"  RGB stems      (first 5)  : {list(rgb_dict.keys())[:5]}")
        print(f"  Mask stems     (first 5)  : {list(mask_dict.keys())[:5]}")
        print("\n  Fix: update the find_file() logic to match your actual filenames.")

    # Show matched samples
    matched = manifest[manifest["has_hdr"]].head(3)
    if len(matched):
        print("\n  Sample matches (image_id → hdr filename):")
        for _, r in matched.iterrows():
            print(f"    {r['image_id']:>6}  →  {Path(str(r['hdr_path'])).name}")

    out = OUTPUT_DIR / "manifest.csv"
    manifest.to_csv(out, index=False)
    print(f"\n  Manifest saved → {out}")
    return manifest


# ══════════════════════════════════════════════
# 3. CLASS DISTRIBUTION
# ══════════════════════════════════════════════
def print_distribution(manifest):
    print("\n" + "="*55)
    print("  3/5  Class distribution")
    print("="*55)

    valid = manifest[manifest["binary_label"].isin([0, 1])]
    print(f"\n  Total images    : {len(manifest)}")
    print(f"  Labelled images : {len(valid)}")

    n = {}
    for label, name in CLASS_NAMES.items():
        n[label] = int((valid["binary_label"] == label).sum())
        pct = 100 * n[label] / len(valid) if len(valid) else 0
        print(f"  {name:22s}: {n[label]:4d}  ({pct:.1f}%)")

    patients = valid.groupby("patient_id")["binary_label"].first()
    print(f"\n  Unique patients : {len(patients)}")
    for label, name in CLASS_NAMES.items():
        print(f"  {name:22s}: {int((patients == label).sum()):4d} patients")

    if n.get(1, 0) > 0:
        ratio = n[0] / n[1]
        print(f"\n  Imbalance ratio (non-cancer : cancer) = {ratio:.1f} : 1")
        if ratio > 2:
            print("  ⚠  Weighted loss + oversampling recommended")

    # Per-diagnosis breakdown
    print(f"\n  Per-diagnosis image counts:")
    print(manifest["diag_raw"].value_counts().to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (counts, title) in zip(axes, [
        ([n[0], n[1]], "Image-level"),
        ([int((patients == 0).sum()), int((patients == 1).sum())], "Patient-level"),
    ]):
        bars = ax.bar(
            ["Non-Cancer", "Cancer (OSCC)"], counts,
            color=[CLASS_COLORS[0], CLASS_COLORS[1]], edgecolor="white"
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Count")
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")
    plt.suptitle("Binary class distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {out}")


# ══════════════════════════════════════════════
# 4. VERIFY ONE SAMPLE
# ══════════════════════════════════════════════
def verify_samples(manifest):
    print("\n" + "="*55)
    print("  4/5  Verifying file shapes")
    print("="*55)

    valid = manifest[
        manifest["has_hdr"] & manifest["has_rgb"] & manifest["has_mask"] &
        manifest["binary_label"].isin([0, 1])
    ]
    if len(valid) == 0:
        print("  SKIPPING — no complete triplets found.")
        print("  Check the manifest.csv to see which files are missing.")
        return

    s = valid.iloc[0]
    print(f"\n  Sample : image_id={s['image_id']}  diag={s['diag_raw']}  label={CLASS_NAMES[int(s['binary_label'])]}")

    # MSI
    try:
        hdr = envi.open(str(s["hdr_path"]))
        msi = hdr.load().astype(np.float32)
        print(f"\n  MSI  shape : {msi.shape}   (expect H×W×16)")
        print(f"  MSI  range : [{msi.min():.4f}, {msi.max():.4f}]")
        try:
            wl = hdr.bands.centers
            print(f"  Wavelengths: {[round(w) for w in wl]} nm")
        except Exception:
            print("  Wavelengths: not stored in header")
    except Exception as e:
        print(f"  MSI  ERROR : {e}")

    # RGB
    try:
        rgb = np.array(Image.open(s["rgb_path"]).convert("RGB"))
        print(f"\n  RGB  shape : {rgb.shape}   (expect H×W×3)")
        print(f"  RGB  range : [{rgb.min()}, {rgb.max()}]")
    except Exception as e:
        print(f"  RGB  ERROR : {e}")

    # Mask
    try:
        mask     = np.array(Image.open(s["mask_path"]).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)
        print(f"\n  Mask shape : {mask.shape}   (expect H×W)")
        print(f"  Mask unique: {np.unique(mask_bin)}")
        print(f"  ROI cover  : {100 * mask_bin.mean():.1f}%")
    except Exception as e:
        print(f"  Mask ERROR : {e}")


# ══════════════════════════════════════════════
# 5. VISUALIZE ONE SAMPLE PER CLASS
# ══════════════════════════════════════════════
def visualize_samples(manifest):
    print("\n" + "="*55)
    print("  5/5  Visualizing samples")
    print("="*55)

    valid = manifest[
        manifest["has_hdr"] & manifest["has_rgb"] & manifest["has_mask"] &
        manifest["binary_label"].isin([0, 1])
    ]
    if len(valid) == 0:
        print("  SKIPPING — no complete triplets.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "One sample per class  |  RGB · Masked RGB · MSI band 8 · Mean spectrum",
        fontsize=12, fontweight="bold"
    )

    for row_idx, label in enumerate([0, 1]):
        rows  = valid[valid["binary_label"] == label]
        if len(rows) == 0:
            continue
        s     = rows.iloc[0]
        name  = CLASS_NAMES[label]
        color = CLASS_COLORS[label]
        ax    = axes[row_idx]
        print(f"\n  {name} → image_id={s['image_id']}  diag={s['diag_raw']}")

        # col 0 — RGB
        try:
            rgb = np.array(Image.open(s["rgb_path"]).convert("RGB"))
            ax[0].imshow(rgb)
            ax[0].set_title(f"{name}\nRGB", color=color, fontweight="bold")
        except Exception as e:
            ax[0].set_title(f"RGB error\n{e}")
        ax[0].axis("off")

        # col 1 — Masked RGB
        try:
            rgb  = np.array(Image.open(s["rgb_path"]).convert("RGB"))
            mask = np.array(Image.open(s["mask_path"]).convert("L"))
            if mask.shape != rgb.shape[:2]:
                mask = np.array(Image.fromarray(mask).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            mbin = (mask > 127).astype(np.float32)
            ax[1].imshow((rgb * mbin[..., None]).astype(np.uint8))
            ax[1].set_title(f"Masked RGB\nROI={100*mbin.mean():.1f}%")
        except Exception as e:
            ax[1].set_title(f"Masked error\n{e}")
        ax[1].axis("off")

        # col 2 — MSI band 8
        try:
            msi  = envi.open(str(s["hdr_path"])).load().astype(np.float32)
            band = msi[:, :, min(8, msi.shape[2] - 1)]
            im   = ax[2].imshow(band, cmap="viridis")
            plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            ax[2].set_title("MSI band 8 (~521nm)")
        except Exception as e:
            ax[2].set_title(f"MSI error\n{e}")
        ax[2].axis("off")

        # col 3 — Mean ROI spectrum
        try:
            hdr  = envi.open(str(s["hdr_path"]))
            msi  = hdr.load().astype(np.float32)
            mask = np.array(Image.open(s["mask_path"]).convert("L"))
            if mask.shape != msi.shape[:2]:
                mask = np.array(Image.fromarray(mask).resize(
                    (msi.shape[1], msi.shape[0]), Image.NEAREST))
            roi    = msi[mask > 127]
            mean_s = roi.mean(axis=0)
            std_s  = roi.std(axis=0)
            try:
                wl = hdr.bands.centers
            except Exception:
                wl = list(range(460, 601, 9))[:16]
            ax[3].plot(wl, mean_s, color=color, linewidth=2)
            ax[3].fill_between(wl, mean_s - std_s, mean_s + std_s, alpha=0.25, color=color)
            ax[3].set_xlabel("Wavelength (nm)", fontsize=9)
            ax[3].set_ylabel("Intensity", fontsize=9)
            ax[3].set_title("Mean ROI spectrum")
            ax[3].grid(True, alpha=0.3)
        except Exception as e:
            ax[3].set_title(f"Spectrum error\n{e}")

    plt.tight_layout()
    out = OUTPUT_DIR / "sample_visualization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved → {out}")


# ══════════════════════════════════════════════
# BONUS — spectral signatures all classes
# ══════════════════════════════════════════════
def plot_all_spectra(manifest):
    print("\n  Plotting spectral signatures per class...")
    colors  = ["#1D9E75", "#888780", "#534AB7", "#185FA5", "#D85A30"]
    classes = [c for c in manifest["diag_raw"].dropna().unique() if str(c) not in ("nan","")]
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for diag, color in zip(classes, colors):
        rows    = manifest[
            (manifest["diag_raw"].astype(str).str.strip() == str(diag).strip()) &
            manifest["has_hdr"] & manifest["has_mask"]
        ].head(5)
        spectra = []
        for _, s in rows.iterrows():
            try:
                hdr  = envi.open(str(s["hdr_path"]))
                msi  = hdr.load().astype(np.float32)
                mask = np.array(Image.open(s["mask_path"]).convert("L"))
                if mask.shape != msi.shape[:2]:
                    mask = np.array(Image.fromarray(mask).resize(
                        (msi.shape[1], msi.shape[0]), Image.NEAREST))
                roi = msi[mask > 127]
                if len(roi):
                    spectra.append(roi.mean(axis=0))
            except Exception:
                continue
        if not spectra:
            continue
        arr  = np.array(spectra)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        try:
            wl = envi.open(str(rows.iloc[0]["hdr_path"])).bands.centers
        except Exception:
            wl = list(range(460, 601, 9))[:16]
        ax.plot(wl, mean, color=color, linewidth=2, label=str(diag))
        ax.fill_between(wl, mean - std, mean + std, alpha=0.15, color=color)
        plotted += 1

    if plotted:
        ax.set_xlabel("Wavelength (nm)", fontsize=11)
        ax.set_ylabel("Normalised intensity", fontsize=11)
        ax.set_title("Mean spectral signature per class (ROI)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = OUTPUT_DIR / "spectral_signatures_all_classes.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Spectral plot saved → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█"*55)
    print("  MODID — DATA AUDIT")
    print("█"*55)

    for name, path in [
        ("BASE_DIR",      BASE_DIR),
        ("PROCESSED_DIR", PROCESSED_DIR),
        ("RGB_DIR",       RGB_DIR),
        ("MASK_DIR",      MASK_DIR),
    ]:
        status = "✓ found" if path.exists() else "✗ NOT FOUND"
        print(f"  {name:15s}: {path}  [{status}]")

    if not BASE_DIR.exists():
        print(f"\n  ERROR: '{BASE_DIR}' not found.")
        sys.exit(1)

    df       = load_labels()
    manifest = build_manifest(df)
    print_distribution(manifest)
    verify_samples(manifest)
    visualize_samples(manifest)
    plot_all_spectra(manifest)

    print("\n" + "="*55)
    print("  AUDIT COMPLETE")
    print(f"  Outputs → {OUTPUT_DIR.resolve()}")
    print("  Next step → step2_preprocessing.py")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
