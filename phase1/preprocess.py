# phase1/preprocess.py
"""
Phase-1: deterministic exact tiling (NO bbox detection, NO shaving).
This file replaces the previous bbox-based preprocess and guarantees
that tiles are exact equal subdivisions of the full input image.

Saves:
  - enhanced.png        (CLAHE + denoise)
  - tiles/tile_r_c.png
  - tiles/tile_r_c_mask.png  (Otsu produced mask, resized to tile)
  - metadata.json
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

from .utils import apply_clahe, denoise, otsu_thresh, morphological_clean

# configuration
MIN_TILE_SIDE = 8
MORPH_KERNEL = 3
MORPH_MIN_AREA = 20

def detect_grid_from_folder(folder_name: str):
    name = folder_name.lower()
    for g in ["2x2", "4x4", "8x8"]:
        if g in name:
            r, c = g.split("x")
            return int(r), int(c)
    return None, None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def preprocess_image(in_path: Path, out_path: Path, rows: int, cols: int):
    """
    Deterministic tiling of the full image into rows x cols, without any cropping.
    """
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_dir(out_path)
    enhanced = apply_clahe(img)
    enhanced = denoise(enhanced, h=8)
    cv2.imwrite(str(out_path / "enhanced.png"), enhanced)

    H, W = enhanced.shape[:2]

    # compute integer cell sizes distributing remainders to the left/top
    base_w = W // cols
    rem_w = W % cols
    cell_ws = [base_w + (1 if c < rem_w else 0) for c in range(cols)]

    base_h = H // rows
    rem_h = H % rows
    cell_hs = [base_h + (1 if r_ < rem_h else 0) for r_ in range(rows)]

    # compute offsets
    x_offsets = [0]
    for w_c in cell_ws[:-1]:
        x_offsets.append(x_offsets[-1] + w_c)
    y_offsets = [0]
    for h_r in cell_hs[:-1]:
        y_offsets.append(y_offsets[-1] + h_r)

    tiles_dir = out_path / "tiles"
    ensure_dir(tiles_dir)

    saved = 0
    filenames = []
    trim_info_map = {}

    for r in range(rows):
        for c in range(cols):
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            # safety clamp
            x1i = max(0, int(round(x1)))
            y1i = max(0, int(round(y1)))
            x2i = min(W, int(round(x2)))
            y2i = min(H, int(round(y2)))

            if x2i <= x1i or y2i <= y1i:
                continue

            tile = enhanced[y1i:y2i, x1i:x2i].copy()
            if tile.shape[0] < MIN_TILE_SIDE or tile.shape[1] < MIN_TILE_SIDE:
                continue

            # produce mask with Otsu on grayscale, then small morphological clean (conservative)
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            mask_tile = otsu_thresh(gray_tile)
            # small clean to remove specks but keep most shape (use conservative params)
            try:
                mask_tile = morphological_clean(mask_tile, kernel_size=MORPH_KERNEL, min_area=MORPH_MIN_AREA)
            except Exception:
                # fallback: keep raw otsu mask
                pass

            # ensure mask size matches tile exactly
            if mask_tile.shape[:2] != tile.shape[:2]:
                mask_tile = cv2.resize(mask_tile, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST)

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            mask_name = f"tile_{r:02d}_{c:02d}_mask.png"
            cv2.imwrite(str(tiles_dir / tile_name), tile)
            cv2.imwrite(str(tiles_dir / mask_name), mask_tile)

            filenames.append(tile_name)
            # trim_info is zeros because we DO NOT trim anything in this version
            trim_info_map[tile_name] = (0,0,0,0)
            saved += 1

    meta = {
        "source": str(in_path),
        "rows": int(rows),
        "cols": int(cols),
        "bbox": [0, 0, W, H],            # full image bbox (explicit)
        "num_tiles_saved": int(saved),
        "tile_size": [int(cell_hs[0]), int(cell_ws[0])],
        "tile_filenames": filenames,
        "trim_info": trim_info_map
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path, meta
