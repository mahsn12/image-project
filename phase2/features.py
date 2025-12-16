# phase2/features.py
# Robust loader, mask sanitization, border/edge similarity helpers.

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import re
import json

# ==================== TILE LOADER ====================

def load_tiles_from_phase1(source_root: str, category: str, identifier: str) -> List[Dict]:
    """
    Load tiles from Phase 1 outputs.
    
    Looks for:
    - tile_{r:02d}_{c:02d}.png (main enhanced tile)
    - tile_{r:02d}_{c:02d}_mask.png (binary mask)
    """
    resource_dir = os.path.join(source_root, category, identifier, "tiles")
    if not os.path.isdir(resource_dir):
        raise FileNotFoundError(f"Tiles directory not found: {resource_dir}")

    # Read metadata if present
    metadata_path = os.path.join(source_root, category, identifier, "metadata.json")
    meta = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except:
            meta = {}

    def _load_one(img_path):
        """Load a single tile image and its mask."""
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        base = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(resource_dir, f"{base}_mask.png")
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        else:
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        
        # Ensure mask is single-channel
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Resize mask to match image if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is 0/255 and sanitize
        mask = (mask > 0).astype(np.uint8) * 255
        # Sanitize masks: morphological clean + remove small components
        # Scale kernel size based on image size
        avg_size = (mask.shape[0] + mask.shape[1]) // 2
        scale = avg_size / 112.0
        kernel_size = max(2, int(3 * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove tiny connected components (scaled by area)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(10, (mask.shape[0] * mask.shape[1]) // 1000)
        clean_mask = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(clean_mask, [c], -1, 255, -1)
        mask = clean_mask
        
        return {"id": base, "img": img, "mask": mask, "path": img_path}

    # 1) Try metadata ordering
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

    # 2) Try numeric pattern tile_{r}_{c}.png
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


# ==================== FEATURE HELPERS ====================

def extract_border_strips(source_img: np.ndarray, source_mask: np.ndarray, fraction: float):
    """Extract 4 border strips from image."""
    height, width = source_img.shape[:2]
    pixel_height = max(1, int(round(height * fraction)))
    pixel_width = max(1, int(round(width * fraction)))
    
    top_start = pixel_height
    left_start = pixel_width
    
    upper_img = source_img[top_start:top_start+pixel_height, :]
    upper_mask = source_mask[top_start:top_start+pixel_height, :]
    lower_img = source_img[height - (pixel_height*2):height-pixel_height, :]
    lower_mask = source_mask[height - (pixel_height*2):height-pixel_height, :]
    start_img = source_img[:, left_start:left_start+pixel_width]
    start_mask = source_mask[:, left_start:left_start+pixel_width]
    end_img = source_img[:, width - (pixel_width*2):width-pixel_width]
    end_mask = source_mask[:, width - (pixel_width*2):width-pixel_width]
    
    # Safety: fallback if any slice is empty
    if upper_img.size == 0:
        upper_img = source_img[0:pixel_height, :]
        upper_mask = source_mask[0:pixel_height, :]
    if lower_img.size == 0:
        lower_img = source_img[height-pixel_height:height, :]
        lower_mask = source_mask[height-pixel_height:height, :]
    if start_img.size == 0:
        start_img = source_img[:, 0:pixel_width]
        start_mask = source_mask[:, 0:pixel_width]
    if end_img.size == 0:
        end_img = source_img[:, width-pixel_width:width]
        end_mask = source_mask[:, width-pixel_width:width]
    
    return upper_img, upper_mask, lower_img, lower_mask, start_img, start_mask, end_img, end_mask


def rms_distance(first_img, first_mask, second_img, second_mask) -> float:
    """Compute RMS distance between two masked images."""
    if first_img.shape[:2] != second_img.shape[:2]:
        min_h = min(first_img.shape[0], second_img.shape[0])
        min_w = min(first_img.shape[1], second_img.shape[1])
        first_img = first_img[:min_h, :min_w]
        second_img = second_img[:min_h, :min_w]
        first_mask = first_mask[:min_h, :min_w]
        second_mask = second_mask[:min_h, :min_w]
    
    valid_mask = (first_mask > 0) & (second_mask > 0)
    if not np.any(valid_mask):
        return 255.0
    
    squared_diff = (first_img.astype(np.float32) - second_img.astype(np.float32)) ** 2
    if squared_diff.ndim == 3:
        squared_diff = squared_diff.mean(axis=2)
    
    squared_diff = squared_diff[valid_mask]
    if squared_diff.size == 0:
        return 255.0
    
    return np.sqrt(np.mean(squared_diff))


def similarity_score(dist) -> float:
    """Convert distance to similarity score [0, 1]."""
    return np.exp(-dist / 50.0)
