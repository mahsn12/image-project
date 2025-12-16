"""
Phase-1: deterministic exact tiling + smart enhancement +
         robust per-tile segmentation & piece isolation.

Saves:
  - enhanced.png
  - tiles/tile_r_c.png          (piece on white background)
  - tiles/tile_r_c_mask.png     (clean binary mask)
  - tiles/tile_r_c_contours.png (debug)
  - metadata.json
"""

from pathlib import Path
import cv2
import numpy as np
import json
import shutil

from .utils import (
    smart_enhance,
    segment_tile,
    isolate_largest_component,
    fill_mask_holes,
    remove_small_specks,
    refine_mask_edges,
    enhance_mask_with_lab,
)

# configuration
MIN_TILE_SIDE = 8
MORPH_KERNEL = 5
MORPH_MIN_AREA = 300
SAVE_DEBUG_CONTOURS = False


def detect_grid_from_folder(folder_name: str):
    name = folder_name.lower()
    for g in ["2x2", "4x4", "8x8"]:
        if g in name:
            r, c = g.split("x")
            return int(r), int(c)
    return None, None


def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def preprocess_image(in_path: Path, out_path: Path, rows: int, cols: int):
    """
    Deterministic tiling: cut raw image first, THEN enhance and segment each tile.
    This prevents enhancement artifacts from affecting tile boundaries.
    """
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_clean_dir(out_path)

    # ---------- 1) Get dimensions from RAW image (no enhancement yet) ----------
    H, W = img.shape[:2]

    # ---------- 2) Compute tile sizes (deterministic grid) ----------
    base_w = W // cols
    rem_w = W % cols
    cell_ws = [base_w + (1 if c < rem_w else 0) for c in range(cols)]

    base_h = H // rows
    rem_h = H % rows
    cell_hs = [base_h + (1 if r < rem_h else 0) for r in range(rows)]

    x_offsets = [0]
    for w_c in cell_ws[:-1]:
        x_offsets.append(x_offsets[-1] + w_c)

    y_offsets = [0]
    for h_r in cell_hs[:-1]:
        y_offsets.append(y_offsets[-1] + h_r)

    tiles_dir = out_path / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    saved = 0

    # ---------- 3) Process each tile: CUT FIRST, then enhance, then segment ----------
    for r in range(rows):
        for c in range(cols):
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            # Cut from RAW image first
            tile_raw = img[y1:y2, x1:x2].copy()
            if tile_raw.shape[0] < MIN_TILE_SIDE or tile_raw.shape[1] < MIN_TILE_SIDE:
                continue
            
            # NOW enhance this individual tile
            # Calculate average tile size for adaptive parameter scaling
            avg_tile_size = (w_tile + h_tile) // 2
            
            tile = smart_enhance(tile_raw, tile_size=avg_tile_size)
            # Save the enhanced tile as the main output (no 'orig' in name)
            # No longer save any '_orig.png' file

            # --- segmentation on the enhanced tile ---
            mask = segment_tile(tile, morph_kernel=MORPH_KERNEL, morph_min_area=MORPH_MIN_AREA, tile_size=avg_tile_size)

            # --- isolate main piece only ---
            mask = isolate_largest_component(mask, min_area=MORPH_MIN_AREA, tile_size=avg_tile_size)

            # --- comprehensive mask refinement pipeline ---
            # 1) Fill holes inside the piece
            mask = fill_mask_holes(mask, min_hole_size=100, tile_size=avg_tile_size)
            
            # 2) Remove small noise specks
            mask = remove_small_specks(mask, min_area=50, tile_size=avg_tile_size)
            
            # 3) Refine edges to avoid hard boundaries
            mask = refine_mask_edges(mask, kernel_size=3, tile_size=avg_tile_size)
            
            # 4) Optional: enhance using LAB for lighting-robust refinement
            mask = enhance_mask_with_lab(tile, mask, tile_size=avg_tile_size)

            # Main output: enhanced tile and refined binary mask

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            mask_name = f"tile_{r:02d}_{c:02d}_mask.png"
            cv2.imwrite(str(tiles_dir / tile_name), tile)
            cv2.imwrite(str(tiles_dir / mask_name), mask)

            filenames.append(tile_name)
            saved += 1

    # ---------- 4) Metadata ----------
    meta = {
        "source": str(in_path),
        "rows": int(rows),
        "cols": int(cols),
        "num_tiles_saved": int(saved),
        "tile_sizes": {
            "row_heights": [int(h) for h in cell_hs],
            "col_widths": [int(w) for w in cell_ws],
        },
        "tile_filenames": filenames,
    }

    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path, meta
