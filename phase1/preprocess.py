"""
Phase-1: deterministic grid tiling that mirrors puzzle_solver.py's core.

We simply cut the image into an exact grid (2x2, 4x4, 8x8 inferred from folder
name), apply a light per-tile enhancement for contrast, and save:
    - tiles/tile_r_c.png      (enhanced tile on white background)
    - tiles/tile_r_c_mask.png (all-white mask, matches solver expectations)
    - metadata.json

No contour segmentation is performed; this aligns preprocessing with the
best-buddies edge matcher used by puzzle_solver.py and the Phase 2 solver.
"""

from pathlib import Path
import cv2
import numpy as np
import json
import shutil

from .utils import smart_enhance, tile_on_white

MIN_TILE_SIDE = 8


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
    Deterministic grid cutting + light enhancement, no contour segmentation.
    Mirrors puzzle_solver.py preprocessing so Phase 2 uses identical cues.
    """

    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_clean_dir(out_path)

    H, W = img.shape[:2]

    # Compute deterministic cell sizes (handles non-divisible dimensions)
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

    for r in range(rows):
        for c in range(cols):
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            tile_raw = img[y1:y2, x1:x2].copy()
            if tile_raw.shape[0] < MIN_TILE_SIDE or tile_raw.shape[1] < MIN_TILE_SIDE:
                continue

            avg_tile_size = (w_tile + h_tile) // 2
            tile_enh = smart_enhance(tile_raw, tile_size=avg_tile_size)

            # White background mask to keep solver expectations consistent
            mask = np.ones(tile_enh.shape[:2], dtype=np.uint8) * 255
            tile_clean = tile_on_white(tile_enh, mask)

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            mask_name = f"tile_{r:02d}_{c:02d}_mask.png"
            cv2.imwrite(str(tiles_dir / tile_name), tile_clean)
            cv2.imwrite(str(tiles_dir / mask_name), mask)

            filenames.append(tile_name)
            saved += 1

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
