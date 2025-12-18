from pathlib import Path
import cv2
import json
import shutil

from .utils import smart_enhance

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
    # Read image and prepare output directory
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_clean_dir(out_path)

    H, W = img.shape[:2]

    # Calculate tile dimensions accounting for remainder pixels
    base_w = W // cols
    rem_w = W % cols
    cell_ws = [base_w + (1 if c < rem_w else 0) for c in range(cols)]

    base_h = H // rows
    rem_h = H % rows
    cell_hs = [base_h + (1 if r < rem_h else 0) for r in range(rows)]

    # Calculate offsets for tile extraction
    x_offsets = [0]
    for w_c in cell_ws[:-1]:
        x_offsets.append(x_offsets[-1] + w_c)

    y_offsets = [0]
    for h_r in cell_hs[:-1]:
        y_offsets.append(y_offsets[-1] + h_r)

    # Extract and save tiles
    tiles_dir = out_path / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    saved = 0

    for r in range(rows):
        for c in range(cols):
            # Extract tile region from image
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            tile_raw = img[y1:y2, x1:x2].copy()
            if tile_raw.shape[0] < MIN_TILE_SIDE or tile_raw.shape[1] < MIN_TILE_SIDE:
                continue

            # Enhance and save tile
            avg_tile_size = (w_tile + h_tile) // 2
            tile_enh = smart_enhance(tile_raw, tile_size=avg_tile_size)

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            cv2.imwrite(str(tiles_dir / tile_name), tile_enh)

            filenames.append(tile_name)
            saved += 1

    # Save metadata with tile information
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
