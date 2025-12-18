import os
import cv2
from typing import List, Dict
import re
import json

def load_tiles_from_phase1(source_root: str, category: str, identifier: str) -> List[Dict]:
    # Load enhanced tiles from Phase 1 output directory
    resource_dir = os.path.join(source_root, category, identifier, "tiles")
    if not os.path.isdir(resource_dir):
        raise FileNotFoundError(f"Tiles directory not found: {resource_dir}")

    # Read metadata if available
    metadata_path = os.path.join(source_root, category, identifier, "metadata.json")
    meta = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except:
            meta = {}

    def _load_one(img_path):
        # Load single tile image with format conversion
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        base = os.path.basename(img_path)[:-4]
        return {"id": base, "img": img, "path": img_path}

    # Try to load tiles using metadata ordering first
    if isinstance(meta, dict) and "tile_filenames" in meta:
        ordered = []
        for fn in meta["tile_filenames"]:
            if not isinstance(fn, str) or not fn.lower().endswith(".png"):
                continue
            if "_mask" in fn or "_contours" in fn or "_inv" in fn:
                continue
            candidate = os.path.join(resource_dir, fn)
            if os.path.exists(candidate):
                item = _load_one(candidate)
                if item is not None:
                    ordered.append(item)
        if len(ordered) > 0:
            return ordered

    # Fall back to numeric pattern matching (tile_r_c.png)
    all_pngs = sorted([f for f in os.listdir(resource_dir) if f.lower().endswith(".png")])
    numeric_pattern = re.compile(r"tile_(\d+)_(\d+)\.png$", flags=re.IGNORECASE)
    
    tiles = []
    for fn in all_pngs:
        m = numeric_pattern.match(fn)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            tiles.append((r, c, fn))
    
    if tiles:
        tiles.sort(key=lambda x: (x[0], x[1]))
        loaded = []
        for r, c, fn in tiles:
            candidate = os.path.join(resource_dir, fn)
            item = _load_one(candidate)
            if item is not None:
                loaded.append(item)
        if loaded:
            return loaded

    raise RuntimeError(f"No tiles found in {resource_dir}")
