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
    extract_contours_from_mask,
    draw_contours_on_image,
)

# configuration
MIN_TILE_SIDE = 8
MORPH_KERNEL = 5
MORPH_MIN_AREA = 300
SAVE_DEBUG_CONTOURS = True


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
    Deterministic tiling of the full image into rows x cols, after smart enhancement.
    Also produces segmented masks per tile and isolates main puzzle piece.
    """
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_clean_dir(out_path)

    # ---------- 1) Enhance full image ----------
    enhanced = smart_enhance(img)
    cv2.imwrite(str(out_path / "enhanced.png"), enhanced)

    H, W = enhanced.shape[:2]

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

    # ---------- 3) Process each tile ----------
    for r in range(rows):
        for c in range(cols):
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            tile = enhanced[y1:y2, x1:x2].copy()
            if tile.shape[0] < MIN_TILE_SIDE or tile.shape[1] < MIN_TILE_SIDE:
                continue
            
            # save the original enhanced tile BEFORE any segmentation or mask
            orig_name = f"tile_{r:02d}_{c:02d}_orig.png"
            cv2.imwrite(str(tiles_dir / orig_name), tile)

            # --- segmentation ---
            mask = segment_tile(tile, morph_kernel=MORPH_KERNEL, morph_min_area=MORPH_MIN_AREA)

            # --- isolate main piece only ---
            mask = isolate_largest_component(mask, min_area=MORPH_MIN_AREA)

            # --- create clean tile (piece on white background) ---
            clean_tile = np.full(tile.shape, 255, dtype=np.uint8)
            clean_tile[mask == 255] = tile[mask == 255]

            # --- optional contour debug image ---
            if SAVE_DEBUG_CONTOURS:
                cnts = extract_contours_from_mask(mask)
                contour_img = draw_contours_on_image(clean_tile, cnts[:1])
            else:
                contour_img = None

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            mask_name = f"tile_{r:02d}_{c:02d}_mask.png"

            cv2.imwrite(str(tiles_dir / tile_name), clean_tile)
            cv2.imwrite(str(tiles_dir / mask_name), mask)

            if contour_img is not None:
                cv2.imwrite(str(tiles_dir / f"tile_{r:02d}_{c:02d}_contours.png"), contour_img)

            filenames.append(tile_name)
            saved += 1

    # ---------- 4) Metadata ----------
    meta = {
        "source": str(in_path),
        "rows": int(rows),
        "cols": int(cols),
        "bbox": [0, 0, W, H],
        "num_tiles_saved": int(saved),
        "tile_sizes": {
            "row_heights": [int(h) for h in cell_hs],
            "col_widths": [int(w) for w in cell_ws],
        },
        "tile_filenames": filenames,
        "trim_info": {name: (0, 0, 0, 0) for name in filenames},  # no trimming yet
    }

    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path, meta
