#!/usr/bin/env python3
"""
evaluate_benchmark.py — minimal evaluator for DeepD3 models

Per case:
  *_stack.tif                (required)
  *_spines.tif               (optional GT)
  *_dendrite.tif             (optional GT)
  *_meta.json                (optional; supports two formats)

Metrics:
  - Recall (scikit-learn)
  - IoU/Jaccard (scikit-learn)
  - NSD (DeepMind surface-distance) with τ = 1 voxel in z

DeepD3 channels:
  prediction[..., 0] -> dendrite
  prediction[..., 1] -> spines
"""

import os, json, argparse
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TF logs

import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from sklearn.metrics import recall_score, jaccard_score
from surface_distance import metrics as surf_metrics

from deepd3.core.analysis import Stack

# --- NumPy <-> >=1.24 compatibility shim (must be first) ---
import numpy as np  # noqa
if not hasattr(np, "int"):     np.int = int         # type: ignore[attr-defined]
if not hasattr(np, "float"):   np.float = float     # type: ignore[attr-defined]
if not hasattr(np, "bool"):    np.bool = bool       # type: ignore[attr-defined]
if not hasattr(np, "object"):  np.object = object   # type: ignore[attr-defined]
if not hasattr(np, "complex"): np.complex = complex # type: ignore[attr-defined]



# --------------------------- metric helpers ---------------------------

def _as_3d_bool(mask):
    """Ensure boolean (Z,Y,X). If 2D, add Z=1."""
    m = (np.asarray(mask) > 0)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3:
        raise ValueError(f"Expected 2D/3D mask; got shape {m.shape}")
    return m.astype(bool)

def compute_recall_iou_sklearn(pred_bin, gt_bin):
    y_true = (np.asarray(gt_bin) > 0).ravel().astype(np.uint8)
    y_pred = (np.asarray(pred_bin) > 0).ravel().astype(np.uint8)
    rec = recall_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    return float(rec), float(iou)

def compute_nsd_deepmind(pred_bin, gt_bin, spacing_mm, tol_z_pixels=1):
    """NSD at tolerance τ = tol_z_pixels * z_spacing (mm)."""
    gt = _as_3d_bool(gt_bin)
    pr = _as_3d_bool(pred_bin)

    # edge cases
    if not gt.any() and not pr.any():
        return 1.0
    if gt.any() ^ pr.any():
        return 0.0

    surf = surf_metrics.compute_surface_distances(gt, pr, spacing_mm)
    tol_mm = float(spacing_mm[0]) * float(tol_z_pixels)  # z-first
    nsd = surf_metrics.compute_surface_dice_at_tolerance(surf, tol_mm)
    return float(nsd)


# --------------------------- inference helpers ---------------------------

def run_inference(stack_path, model_path, mode, tile, inset, pad_op_name, use_gpu=True):
    """
    Runs DeepD3 inference and returns prediction volume:
      shape (Z, Y, X, C), channels: [dendrite, spines, ...]
    """
    S = Stack(str(stack_path))
    pad_op = np.mean if pad_op_name == "mean" else np.min
    device = "/gpu:0" if use_gpu and tf.config.list_physical_devices('GPU') else "/cpu:0"

    with tf.device(device):
        if mode == "tile4":
            S.predictFourFold(model_path, tile_size=tile, inset_size=inset, pad_op=pad_op)
        elif mode == "tile1":
            S.predictInset(model_path, tile, inset)
        else:
            S.predictWholeImage(model_path)

    if not getattr(S, "segmented", False):
        raise RuntimeError("Inference produced no prediction (S.segmented is False).")
    return S.prediction


# --------------------------- dataset helpers ---------------------------

def find_cases(root: Path):
    for stack_path in sorted(root.glob("**/*_stack.tif")):
        base = str(stack_path)[:-len("_stack.tif")]
        yield {
            "name": Path(base).name,
            "stack": stack_path,
            "sp_gt": Path(base + "_spines.tif"),
            "de_gt": Path(base + "_dendrite.tif"),
            "meta":  Path(base + "_meta.json"),
        }

def load_spacing_mm(meta_path: Path, default_xy_um: float, default_z_um: float):
    """
    Returns:
      spacing_mm: (z_mm, y_mm, x_mm)
      xy_um: representative in-plane pixel size, in µm (uses Y)
      z_um: axial step, in µm

    Supports:
      A) {"xy": <µm>, "z": <µm>}
      B) {"pixel_sizes": {"sizes": [z_m, y_m, x_m]}, ...}  # meters
    """
    xy_um, z_um = default_xy_um, default_z_um

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Format A
            if "xy" in meta or "z" in meta:
                xy_um = float(meta.get("xy", xy_um))
                z_um  = float(meta.get("z",  z_um))
                spacing_mm = (z_um/1000.0, xy_um/1000.0, xy_um/1000.0)

            # Format B (meters, order [Z,Y,X])
            elif "pixel_sizes" in meta and "sizes" in meta["pixel_sizes"]:
                sizes_m = meta["pixel_sizes"]["sizes"]
                if not (isinstance(sizes_m, (list, tuple)) and len(sizes_m) == 3):
                    raise ValueError("pixel_sizes.sizes must be length-3 [z_m,y_m,x_m]")
                z_m, y_m, x_m = map(float, sizes_m)
                spacing_mm = (z_m*1000.0, y_m*1000.0, x_m*1000.0)  # m -> mm
                z_um = z_m * 1e6
                xy_um = y_m * 1e6  # representative in-plane (assumes x≈y)

            else:
                # Unknown format -> defaults
                spacing_mm = (z_um/1000.0, xy_um/1000.0, xy_um/1000.0)

        except Exception:
            spacing_mm = (z_um/1000.0, xy_um/1000.0, xy_um/1000.0)
    else:
        spacing_mm = (z_um/1000.0, xy_um/1000.0, xy_um/1000.0)

    return spacing_mm, xy_um, z_um


# --------------------------- main script ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate DeepD3 with Recall/IoU (scikit) and NSD (DeepMind, τ=1 z-voxel).")
    ap.add_argument("--data", required=True, help="Root with *_stack.tif (+ *_spines.tif, *_dendrite.tif, *_meta.json).")
    ap.add_argument("--model", required=True, help="Path to Keras .h5 model.")
    ap.add_argument("--out", default="eval_out", help="Output folder for CSVs.")
    ap.add_argument("--mode", choices=["plane","tile1","tile4"], default="plane", help="Inference mode.")
    ap.add_argument("--tile", type=int, default=128, help="Tile size (tile modes).")
    ap.add_argument("--inset", type=int, default=96, help="Inset size (tile modes).")
    ap.add_argument("--pad-op", choices=["mean","min"], default="mean", help="Padding op for tile4.")
    ap.add_argument("--th-sp", type=float, default=0.5, help="Threshold for spines binarization.")
    ap.add_argument("--th-de", type=float, default=0.5, help="Threshold for dendrite binarization.")
    ap.add_argument("--xy-um", type=float, default=0.094, help="Fallback XY pixel size in µm.")
    ap.add_argument("--z-um",  type=float, default=0.5,   help="Fallback Z step in µm.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (otherwise auto GPU if available).")
    args = ap.parse_args()

    root = Path(args.data)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in find_cases(root):
        name = case["name"]
        try:
            # Inference
            pred = run_inference(case["stack"], args.model, args.mode, args.tile, args.inset, args.pad_op, use_gpu=not args.cpu)
            de_prob = pred[..., 0]
            sp_prob = pred[..., 1]

            # Spacing from meta (supports both formats)
            spacing_mm, xy_um, z_um = load_spacing_mm(case["meta"], args.xy_um, args.z_um)

            row = {"case": name, "xy_um": xy_um, "z_um": z_um}

            # Evaluate SPINES if GT exists
            if case["sp_gt"].exists():
                sp_gt = tiff.imread(str(case["sp_gt"]))
                sp_bin = (sp_prob >= args.th_sp).astype(np.uint8)
                sp_recall, sp_iou = compute_recall_iou_sklearn(sp_bin, sp_gt)
                sp_nsd = compute_nsd_deepmind(sp_bin, sp_gt, spacing_mm, tol_z_pixels=1)
                row.update(dict(sp_recall=sp_recall, sp_iou=sp_iou, sp_nsd=sp_nsd))
            else:
                row.update(dict(sp_recall=np.nan, sp_iou=np.nan, sp_nsd=np.nan))

            # Evaluate DENDRITE if GT exists
            if case["de_gt"].exists():
                de_gt = tiff.imread(str(case["de_gt"]))
                de_bin = (de_prob >= args.th_de).astype(np.uint8)
                de_recall, de_iou = compute_recall_iou_sklearn(de_bin, de_gt)
                de_nsd = compute_nsd_deepmind(de_bin, de_gt, spacing_mm, tol_z_pixels=1)
                row.update(dict(de_recall=de_recall, de_iou=de_iou, de_nsd=de_nsd))
            else:
                row.update(dict(de_recall=np.nan, de_iou=np.nan, de_nsd=np.nan))

            rows.append(row)

            print(f"[OK] {name}  "
                  f"SP(r={row['sp_recall']:.3f}, IoU={row['sp_iou']:.3f}, NSD={row['sp_nsd']:.3f})  "
                  f"DE(r={row['de_recall']:.3f}, IoU={row['de_iou']:.3f}, NSD={row['de_nsd']:.3f})")

        except Exception as e:
            print(f"[ERR] {name}: {e}")
            rows.append({"case": name, "xy_um": np.nan, "z_um": np.nan,
                         "sp_recall": np.nan, "sp_iou": np.nan, "sp_nsd": np.nan,
                         "de_recall": np.nan, "de_iou": np.nan, "de_nsd": np.nan,
                         "error": str(e)})

    # Per-case CSV
    df = pd.DataFrame(rows)
    per_case_csv = out_dir / "results.csv"
    df.to_csv(per_case_csv, index=False)
    print(f"Wrote {per_case_csv}")

    # Mean ± std summary
    metrics = ["sp_recall","sp_iou","sp_nsd","de_recall","de_iou","de_nsd"]
    stats = []
    for m in metrics:
        vals = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(vals):
            stats.append({"metric": m, "mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n": int(len(vals))})
    stats_df = pd.DataFrame(stats)
    summary_csv = out_dir / "results_summary.csv"
    stats_df.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

if __name__ == "__main__":
    main()
