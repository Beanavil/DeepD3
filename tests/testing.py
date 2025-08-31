#!/usr/bin/env python3
import argparse, json, os, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd

# use the same “engine” the GUI uses
from deepd3.core.analysis import Stack, ROI2D_Creator, ROI3D_Creator
from scipy.ndimage import gaussian_filter

# I/O that tolerates TIFF stacks
try:
    import tifffile as tiff
    def read_tiff(p): return tiff.imread(str(p))
    def write_tiff(p, arr): 
        p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(str(p), arr)
except ImportError:
    import imageio.v3 as iio
    def read_tiff(p): return iio.imread(str(p))
    def write_tiff(p, arr): 
        p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(p), arr)

def binarize(x):
    x = np.asarray(x)
    return (x > 0).astype(np.uint8)

def metrics(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt   = (np.asarray(gt)   > 0).astype(np.uint8)

    tp = np.logical_and(pred==1, gt==1).sum()
    tn = np.logical_and(pred==0, gt==0).sum()
    fp = np.logical_and(pred==1, gt==0).sum()
    fn = np.logical_and(pred==0, gt==1).sum()

    precision = tp / (tp + fp) if (tp+fp) else 0.0
    recall    = tp / (tp + fn) if (tp+fn) else 0.0
    iou       = tp / (tp + fp + fn) if (tp+fp+fn) else 0.0
    dice      = (2*tp) / (2*tp + fp + fn) if (2*tp+fp+fn) else 0.0
    f1        = (2*precision*recall) / (precision+recall) if (precision+recall) else 0.0
    acc       = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
    specificity = tn / (tn + fp) if (tn+fp) else 0.0
    bal_acc   = 0.5 * (recall + specificity)

    return dict(
        precision=precision, recall=recall,
        iou=iou, dice=dice, f1=f1,
        accuracy=acc, balanced=bal_acc
    )

def find_cases(root: Path):
    cases = []
    for stack_path in sorted(root.glob("**/*_stack.tif")):
        base = str(stack_path)[:-len("_stack.tif")]
        cases.append({
            "base": base,
            "stack": stack_path,
            "meta":  Path(base + "_meta.json"),
            "sp_gt": Path(base + "_spines.tif"),
            "de_gt": Path(base + "_dendrite.tif"),
        })
    return cases

def segment_with_stack(S: Stack, model_path: str, mode: str, tile_size: int, inset_size: int, padding_op: str, device: str):
    import tensorflow as tf  # lazy import just like GUI
    pad_op = np.mean if padding_op == "mean" else np.min

    # map the GUI’s three options
    # 'plane' == Whole image (per plane)
    # 'tile1' == tiles 1x
    # 'tile4' == tiles 4x avg
    with tf.device(device):
        if mode == "tile4":
            S.predictFourFold(model_path, tile_size=tile_size, inset_size=inset_size, pad_op=pad_op)
        elif mode == "plane":
            S.predictWholeImage(model_path)
        else:  # "tile1"
            S.predictInset(model_path, tile_size, inset_size)

def maybe_clean(S: Stack, closing: bool, closing_iter: int,
                clean_dendrite: bool, cd_thresh: float, min_d_size: int,
                clean_spines: bool, dend_dilate: int):
    # mirrors Main.cleaning() + previewCleaning() logic
    if closing:
        d = S.closing(int(closing_iter))
        S.prediction[..., 0] = d
        S.prediction[..., 2] = d

    if clean_dendrite:
        d = S.cleanDendrite3D(float(cd_thresh), int(min_d_size))
        S.prediction[..., 0] = d
        S.prediction[..., 2] = d

    if clean_spines:
        s = S.cleanSpines(float(cd_thresh), int(dend_dilate))
        S.prediction[..., 1] = s

def build_roi2d(S: Stack, threshold: float, apply_watershed: bool,
                maxD: int, min_size: int, dend_thresh: float):
    r = ROI2D_Creator(S.prediction[...,0], S.prediction[...,1], float(threshold))
    r.create(apply_watershed)
    # optional cleaning like GUI
    old_n = sum(len(rz) for rz in r.rois.values())
    if maxD is not None and min_size is not None and dend_thresh is not None:
        _, _ = r.clean(int(maxD), int(min_size), float(dend_thresh))
    return r

def build_roi3d(S: Stack, method: str, area_th: float, peak_th: float,
                seed_delta: float, dist_to_seed_um: float, min_px: int, max_px: int, min_planes: int, watershed: bool,
                dimensions: dict):
    # The GUI uses ROI3D_Creator(d, s, method, area, peak, seedDelta, distanceToSeed, dimensions=...)
    r = ROI3D_Creator(S.prediction[...,0], S.prediction[...,1],
                      method, float(area_th), float(peak_th),
                      float(seed_delta), float(dist_to_seed_um),
                      dimensions=dimensions)
    r.create(int(min_px), int(max_px), int(min_planes), bool(watershed))
    return r

def run_case(case, args, out_root: Path):
    stack_path = case["stack"]
    base_name  = Path(case["base"]).name
    out_dir    = out_root / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # dimensions (xy, z) in µm — from meta.json if present, else CLI defaults
    dims = dict(xy=args.xy_um, z=args.z_um)
    if case["meta"].exists():
        try:
            with open(case["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            dims["xy"] = float(meta.get("xy", dims["xy"]))
            dims["z"]  = float(meta.get("z",  dims["z"]))
        except Exception as e:
            print(f"[{base_name}] Warning: failed to parse meta: {e}")

    # Stack class needs a path for predictions file; we point it into out_dir
    pred_fn = str(out_dir / f"{base_name}.prediction")

    # load + segment
    S = Stack(str(stack_path), pred_fn, dims)

    device = "/cpu:0" if args.cpu else "/gpu:0"
    segment_with_stack(S, args.model, args.mode, args.tile, args.inset, args.pad_op, device)

    # optional cleaning (matches GUI toggles)
    maybe_clean(S,
        closing=args.closing,
        closing_iter=args.closing_iter,
        clean_dendrite=args.clean_dendrite,
        cd_thresh=args.clean_dendrite_thresh,
        min_d_size=args.min_dendrite_size,
        clean_spines=args.clean_spines,
        dend_dilate=args.dendrite_dilate,
    )

    # save predictions like the GUI's ExportPredictions would (two channels)
    # channel order in S.prediction: [..., 0]=dendrite, [...,1]=spines, [...,2] may mirror dendrite
    write_tiff(out_dir / f"{base_name}_pred_spines.tif",   S.prediction[...,1].astype(np.float32))
    write_tiff(out_dir / f"{base_name}_pred_dendrite.tif", S.prediction[...,0].astype(np.float32))

    # build ROIs if requested
    roi_summary = {}
    roi_map = None

    if args.roi2d:
        r2 = build_roi2d(S, args.roi2d_thresh, args.roi2d_watershed,
                         args.roi2d_maxD, args.roi2d_min_size, args.roi2d_dend_thresh)
        roi_map = r2.roi_map
        roi_summary["roi2d_count"] = sum(len(v) for v in r2.rois.values())
        write_tiff(out_dir / f"{base_name}_roi2d_map.tif", roi_map.astype(np.uint16))

    if args.roi3d:
        r3 = build_roi3d(S, args.roi3d_method, args.roi3d_area,
                         args.roi3d_peak, args.roi3d_seed_delta, args.roi3d_dist_um,
                         args.roi3d_min_px, args.roi3d_max_px, args.roi3d_min_planes, args.roi3d_watershed, dims)
        roi_map = r3.roi_map
        roi_summary["roi3d_count"] = int((roi_map > 0).max())  # labels up to N
        write_tiff(out_dir / f"{base_name}_roi3d_map.tif", roi_map.astype(np.uint16))

    # evaluate if GT available
    row = {"case": base_name, "xy_um": dims["xy"], "z_um": dims["z"], **roi_summary}
    if case["sp_gt"].exists():
        sp_gt = read_tiff(case["sp_gt"])
        row.update({f"sp_{k}": v for k, v in metrics(S.prediction[...,1], sp_gt).items()})
    if case["de_gt"].exists():
        de_gt = read_tiff(case["de_gt"])
        row.update({f"de_{k}": v for k, v in metrics(S.prediction[...,0], de_gt).items()})

    # quick middle-slice overlay for QC
    def make_overlay(gray, mask, alpha=0.35):
        g = gray.astype(np.float32)
        g = g / g.max() if g.max() > 0 else g
        g = np.clip(g, 0, 1)
        m = (mask > 0).astype(np.float32)
        rgb = np.stack([g, g, g], axis=-1)
        rgb[..., 0] = np.clip(rgb[..., 0] + alpha * m, 0, 1)
        return (rgb * 255).astype(np.uint8)

    z = S.prediction.shape[0] // 2
    try:
        raw = read_tiff(stack_path)
        raw_z = raw[z] if raw.ndim == 3 else raw
        ov_sp = make_overlay(raw_z, S.prediction[z,...,1])
        write_tiff(out_dir / f"{base_name}_overlay_spines.png", ov_sp)
        ov_de = make_overlay(raw_z, S.prediction[z,...,0])
        write_tiff(out_dir / f"{base_name}_overlay_dendrite.png", ov_de)
    except Exception as e:
        print(f"[{base_name}] overlay failed: {e}")

    # write per-case settings for reproducibility
    with open(out_dir / f"{base_name}_runmeta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dimensions_um": dims,
            "model": args.model,
            "mode": args.mode,
            "tile": args.tile,
            "inset": args.inset,
            "pad_op": args.pad_op,
            "cpu": args.cpu,
            "cleaning": {
                "closing": args.closing,
                "closing_iter": args.closing_iter,
                "clean_dendrite": args.clean_dendrite,
                "clean_dendrite_thresh": args.clean_dendrite_thresh,
                "min_dendrite_size": args.min_dendrite_size,
                "clean_spines": args.clean_spines,
                "dendrite_dilate": args.dendrite_dilate,
            },
            "roi2d": {
                "enabled": args.roi2d,
                "threshold": args.roi2d_thresh,
                "watershed": args.roi2d_watershed,
                "maxD": args.roi2d_maxD,
                "min_size": args.roi2d_min_size,
                "dendrite_thresh": args.roi2d_dend_thresh,
            },
            "roi3d": {
                "enabled": args.roi3d,
                "method": args.roi3d_method,
                "area": args.roi3d_area,
                "peak": args.roi3d_peak,
                "seed_delta": args.roi3d_seed_delta,
                "dist_um": args.roi3d_dist_um,
                "min_px": args.roi3d_min_px,
                "max_px": args.roi3d_max_px,
                "min_planes": args.roi3d_min_planes,
                "watershed": args.roi3d_watershed,
            }
        }, f, indent=2)

    return row

def main():
    ap = argparse.ArgumentParser(description="Batch runner for DeepD3 GUI pipeline (turtle benchmarking).")
    ap.add_argument("--data", required=True, help="Path to BIMAP-Rumessa/benchmarking/turtle")
    ap.add_argument("--out", default="out_turtle", help="Output folder")
    ap.add_argument("--model", required=True, help="Path to TF/Keras .h5 model")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default GPU if available)")
    # inference mode & tiling (matches GUI)
    ap.add_argument("--mode", choices=["plane","tile1","tile4"], default="plane",
                    help="plane=Whole image per plane; tile1=Tile inference; tile4=Tile inference 4x average")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--inset", type=int, default=96)
    ap.add_argument("--pad-op", dest="pad_op", choices=["mean","min"], default="mean")

    # default dimensions if meta.json missing
    ap.add_argument("--xy-um", type=float, default=0.094)
    ap.add_argument("--z-um",  type=float, default=0.5)

    # cleaning toggles (match GUI defaults reasonably)
    ap.add_argument("--closing", action="store_true")
    ap.add_argument("--closing-iter", type=int, default=3)
    ap.add_argument("--clean-dendrite", action="store_true")
    ap.add_argument("--clean-dendrite-thresh", type=float, default=0.7)
    ap.add_argument("--min-dendrite-size", type=int, default=100)
    ap.add_argument("--clean-spines", action="store_true")
    ap.add_argument("--dendrite-dilate", type=int, default=21)

    # ROI2D options (GUI: ROI2D dialog)
    ap.add_argument("--roi2d", action="store_true")
    ap.add_argument("--roi2d-thresh", type=float, default=0.25)
    ap.add_argument("--roi2d-watershed", action="store_true")
    ap.add_argument("--roi2d-maxD", type=int, default=30)
    ap.add_argument("--roi2d-min-size", type=int, default=10)
    ap.add_argument("--roi2d-dend-thresh", type=float, default=0.7)

    # ROI3D options (GUI: ROI3D dialog defaults)
    ap.add_argument("--roi3d", action="store_true")
    ap.add_argument("--roi3d-method", choices=["floodfill","connected components"], default="floodfill")
    ap.add_argument("--roi3d-area", type=float, default=0.25)
    ap.add_argument("--roi3d-peak", type=float, default=0.80)
    ap.add_argument("--roi3d-seed-delta", type=float, default=0.2)
    ap.add_argument("--roi3d-dist-um", type=float, default=2.0)
    ap.add_argument("--roi3d-min-px", type=int, default=20)
    ap.add_argument("--roi3d-max-px", type=int, default=1000)
    ap.add_argument("--roi3d-min-planes", type=int, default=1)
    ap.add_argument("--roi3d-watershed", action="store_true")

    args = ap.parse_args()
    root = Path(args.data)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    cases = find_cases(root)
    if not cases:
        print("No *_stack.tif cases found under", root, file=sys.stderr)
        sys.exit(2)

    # run serially (GPU models usually want single-process)
    rows = []
    for c in cases:
        try:
            rows.append(run_case(c, args, out_root))
        except Exception as e:
            rows.append({"case": Path(c["base"]).name, "error": str(e)})
            print(f"[{Path(c['base']).name}] ERROR: {e}", file=sys.stderr)

    # save summary
    csv_path = out_root / "summary.csv"
    # collect keys
    keys = set()
    for r in rows: keys.update(r.keys())
    keys = ["case"] + sorted(k for k in keys if k != "case")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("Wrote", csv_path)

    df = pd.DataFrame(rows)
    # split spines / dendrite metrics
    metrics_cols = [c for c in df.columns if c not in ("case","error")]
    stats = []
    for prefix in ["sp","de"]:
        cols = [c for c in metrics_cols if c.startswith(prefix)]
        for c in cols:
            vals = df[c].dropna().astype(float)
            if len(vals):
                stats.append(dict(
                    metric=c,
                    mean=vals.mean(),
                    std=vals.std()
                ))
    stats_df = pd.DataFrame(stats)
    stats_csv = out_root / "summary_stats.csv"
    stats_df.to_csv(stats_csv, index=False)


if __name__ == "__main__":
    main()
