import cv2
import numpy as np
from pathlib import Path


def split_into_tiles(img, rows, cols):
    """Split image into (rows x cols) tiles."""
    H, W = img.shape[:2]
    tile_h = H // rows
    tile_w = W // cols

    tiles = []
    coords = []

    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_h
            x0 = c * tile_w
            y1 = (r + 1) * tile_h if r < rows - 1 else H
            x1 = (c + 1) * tile_w if c < cols - 1 else W

            tile = img[y0:y1, x0:x1].copy()
            tiles.append(tile)
            coords.append((r, c, y0, x0, y1, x1))

    return tiles, coords


def save_tiles(tiles, out_dir: Path):
    """Save each tile to out_dir/tile_rC_cC.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for idx, tile in enumerate(tiles):
        r = idx // int(len(tiles)**0.5)
        c = idx % int(len(tiles)**0.5)
        fname = out_dir / f"tile_r{r}_c{c}.png"
        cv2.imwrite(str(fname), tile)
        paths.append(str(fname))

    return paths
