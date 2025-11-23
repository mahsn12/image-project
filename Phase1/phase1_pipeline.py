#!/usr/bin/env python3
"""
phase1_full_pipeline.py

Full Phase 1 preprocessing pipeline for mixed dataset (original images + scrambled grids).
Usage examples:
  python phase1_full_pipeline.py --dataset dataset_images --output phase1_outputs
  python phase1_full_pipeline.py -d dataset_images -o phase1_outputs --skip-grids
  python phase1_full_pipeline.py -d dataset_images -o phase1_outputs --no-watershed
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse

# ---------------------
# Configurable defaults
# ---------------------
DEFAULT_DATASET_ROOT = "dataset_images"
DEFAULT_OUTPUT_ROOT = "phase1_outputs"

# module-level dataset_root used by helper functions (set by run_dataset)
dataset_root = None

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Image processing defaults (tweak if necessary)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ---------------------
# Utility functions
# ---------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def is_split_grid_folder(p: Path) -> bool:
    """Heuristic: folder name contains common grid tokens."""
    name = p.name.lower()
    tokens = ["2x2", "2_x_2", "2by2", "4x4", "4_x_4", "4by4", "8x8", "8_x_8", "8by8"]
    return any(t in name for t in tokens)

# ---------------------
# Low-level image ops
# ---------------------
def color_normalize_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def denoise_image(img_bgr):
    return cv2.bilateralFilter(img_bgr, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

def edge_enhance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(gray, 1.5, g, -0.5, 0)
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

# ---------------------
# Segmentation helpers
# ---------------------
def compute_mask_adaptive(img_bgr, min_area=1000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Adaptive parameters scale with image size
    block = 51 if max(h,w) > 400 else 21
    C = 8 if max(h,w) > 400 else 5
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, C)
    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
                              iterations=1)
    # Keep connected components above min_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask

def watershed_split(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return mask
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    fake = cv2.cvtColor((mask > 0).astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(fake, markers)
    split_mask = np.zeros_like(mask)
    for m in range(2, num_markers + 2):
        split_mask[markers == m] = 255
    split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return split_mask

def extract_contours(mask, min_area=1000):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]

def save_cropped_pieces(img, contours, out_dir, prefix="piece"):
    ensure_dir(out_dir)
    pieces = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        pad = int(0.05 * max(w, h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
        crop = img[y0:y1, x0:x1].copy()
        fname = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        cv2.imwrite(fname, crop)
        pieces.append({"index": i, "bbox": [int(x0), int(y0), int(x1-x0), int(y1-y0)], "file": fname})
    return pieces

# ---------------------
# Grid utilities
# ---------------------
def split_image_to_grid(img, rows, cols):
    H, W = img.shape[:2]
    tile_h = H // rows
    tile_w = W // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_h
            x0 = c * tile_w
            y1 = (r+1)*tile_h if r < rows-1 else H
            x1 = (c+1)*tile_w if c < cols-1 else W
            tiles.append((img[y0:y1, x0:x1].copy(), (r, c, x0, y0, x1-x0, y1-y0)))
    return tiles

def grid_size_from_folder(path: Path):
    """Attempt to infer grid size from folder name (e.g., puzzle_2x2 -> (2,2))."""
    name = path.name.lower()
    if "2x2" in name or "2_x_2" in name or "2by2" in name:
        return (2,2)
    if "4x4" in name or "4_x_4" in name or "4by4" in name:
        return (4,4)
    if "8x8" in name or "8_x_8" in name or "8by8" in name:
        return (8,8)
    return None

# ---------------------
# Per-image pipelines
# ---------------------
def process_image_path(img_path: Path, out_root: Path, use_watershed=True):
    """Process a full image (not grid-split). Save artifacts under out_root/<relative parent>/<stem>/"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return None

    rel = img_path.relative_to(dataset_root)
    out_dir = out_root / rel.parent / img_path.stem
    ensure_dir(out_dir)

    cv2.imwrite(str(out_dir / f"{img_path.stem}_orig.png"), img)

    h,w = img.shape[:2]
    image_area = h * w
    min_area = max(300, int(image_area / 2000))  # heuristic; adjust divisor to tune

    img_clahe = color_normalize_clahe(img)
    img_dn = denoise_image(img_clahe)
    img_sharp = edge_enhance(img_dn)

    cv2.imwrite(str(out_dir / f"{img_path.stem}_clahe.png"), img_clahe)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_denoised.png"), img_dn)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_sharp.png"), img_sharp)

    mask = compute_mask_adaptive(img_dn, min_area=min_area)

    # If extremely large single component or too few components, try watershed
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if use_watershed and (num_labels <= 2 or stats[:, cv2.CC_STAT_AREA].max() > 0.5 * image_area):
            m_ws = watershed_split(mask)
            if m_ws.sum() > 0:
                mask = m_ws
    except Exception:
        pass

    cv2.imwrite(str(out_dir / f"{img_path.stem}_mask.png"), mask)

    contours = extract_contours(mask, min_area=min_area)

    overlay = img.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_overlay.png"), overlay)

    pieces_dir = out_dir / "pieces"
    pieces_meta = save_cropped_pieces(img, contours, str(pieces_dir), prefix=img_path.stem + "_piece")

    summary = {
        "source": str(img_path),
        "image_shape": [int(h), int(w)],
        "min_area_used": int(min_area),
        "detected_pieces": len(contours),
        "pieces": pieces_meta
    }
    with open(out_dir / f"{img_path.stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {img_path} -> detected {len(contours)} pieces")
    return summary

def process_grid_image(img_path: Path, out_root: Path, rows=None, cols=None, do_preprocess_tile=True):
    """
    Split grid image into rows x cols tiles (or infer), save tiles and optionally run per-tile preprocessing.
    Artifacts saved under out_root/<relative parent>/<stem>/
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return None

    # infer grid size from folder if not provided
    if rows is None or cols is None:
        inf = grid_size_from_folder(img_path.parent)
        if inf:
            rows, cols = inf
        else:
            # fallback heuristic: try 2/4/8 choose smallest tile-mean variance
            def score_k(k):
                tiles = split_image_to_grid(img, k, k)
                means = [np.mean(cv2.cvtColor(t[0], cv2.COLOR_BGR2GRAY)) for t in tiles]
                return np.std(means)
            candidates = [2,4,8]
            scores = {k: score_k(k) for k in candidates}
            rows = cols = min(scores, key=scores.get)

    rel = img_path.relative_to(dataset_root)
    out_dir = out_root / rel.parent / img_path.stem
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_orig.png"), img)

    tiles = split_image_to_grid(img, rows, cols)
    tiles_dir = out_dir / "pieces"
    ensure_dir(tiles_dir)

    pieces_meta = []
    for tile, (r,c,x,y,w,h) in tiles:
        tile_name = f"{img_path.stem}_tile_r{r}c{c}.png"
        tile_path = tiles_dir / tile_name
        cv2.imwrite(str(tile_path), tile)

        if do_preprocess_tile:
            # tune min_area for tile
            ht, wt = tile.shape[:2]
            tile_min_area = max(50, int((ht * wt) / 200))  # adjust divisor to tune sensitivity
            tile_clahe = color_normalize_clahe(tile)
            tile_dn = denoise_image(tile_clahe)
            tile_sharp = edge_enhance(tile_dn)
            tile_mask = compute_mask_adaptive(tile_dn, min_area=tile_min_area)

            cv2.imwrite(str(out_dir / f"{tile_path.stem}_clahe.png"), tile_clahe)
            cv2.imwrite(str(out_dir / f"{tile_path.stem}_denoised.png"), tile_dn)
            cv2.imwrite(str(out_dir / f"{tile_path.stem}_sharp.png"), tile_sharp)
            cv2.imwrite(str(out_dir / f"{tile_path.stem}_mask.png"), tile_mask)

            tile_contours = extract_contours(tile_mask, min_area=tile_min_area)
            ov = tile.copy()
            if tile_contours:
                cv2.drawContours(ov, tile_contours, -1, (0,255,0), 1)
            cv2.imwrite(str(out_dir / f"{tile_path.stem}_overlay.png"), ov)

        pieces_meta.append({
            "tile_file": str(tile_path),
            "grid_pos": [int(r), int(c)],
            "bbox_in_source": [int(x), int(y), int(w), int(h)]
        })

    summary = {
        "source": str(img_path),
        "grid": [int(rows), int(cols)],
        "n_tiles": len(tiles),
        "tiles_dir": str(tiles_dir),
        "tiles": pieces_meta
    }
    with open(out_dir / f"{img_path.stem}_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {img_path} -> split into {rows}x{cols} = {len(tiles)} tiles")
    return summary

# ---------------------
# Dataset runner
# ---------------------
def run_dataset(dataset_root_str: str, output_root: str, skip_grids=False, use_watershed=True, preprocess_tiles=True):
    global dataset_root  # used in process functions
    dataset_root = Path(dataset_root_str)
    out_root = Path(output_root)
    ensure_dir(out_root)

    all_images = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not all_images:
        raise SystemExit(f"No images found under {dataset_root}")

    summaries = []
    print(f"[INFO] Found {len(all_images)} images under {dataset_root}")
    for p in sorted(all_images):
        # If the folder is detected as a grid folder, treat specially
        if is_split_grid_folder(p.parent):
            if skip_grids:
                print(f"[SKIP] {p} (grid folder skipped)")
                continue
            s = process_grid_image(p, out_root, rows=None, cols=None, do_preprocess_tile=preprocess_tiles)
            if s:
                summaries.append(s)
        else:
            s = process_image_path(p, out_root, use_watershed=use_watershed)
            if s:
                summaries.append(s)

    # Save global summary
    with open(out_root / "summary_all.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("[DONE] All processing complete. Outputs saved to:", out_root)
    return summaries

# ---------------------
# CLI entrypoint
# ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 full pipeline for jigsaw puzzle dataset (grids + originals).")
    p.add_argument("--dataset", "-d", default=DEFAULT_DATASET_ROOT, help="Root dataset folder (e.g., dataset_images)")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_ROOT, help="Output root folder")
    p.add_argument("--skip-grids", dest="skip_grids", action="store_true", help="Skip grid folders like puzzle_2x2/4x4/8x8")
    p.add_argument("--no-watershed", dest="use_watershed", action="store_false", help="Disable watershed splitting")
    p.add_argument("--no-preprocess-tiles", dest="preprocess_tiles", action="store_false", help="Do not run preprocessing on split tiles (just save tiles)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_dataset(args.dataset, args.output, skip_grids=args.skip_grids,
                use_watershed=args.use_watershed, preprocess_tiles=args.preprocess_tiles)

    print("Done.")
