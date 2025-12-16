import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


# ==========================
# SMART ENHANCEMENT PIPELINE
# ==========================

def smart_enhance(img: np.ndarray, tile_size: int = None) -> np.ndarray:
    """
    Edge-friendly enhancement pipeline:
        1) Bilateral denoise to preserve contours
        2) CLAHE on luminance for local contrast
        3) Guided filter (or bilateral fallback) to smooth without blur
        4) Soft unsharp mask for crisp edges
        5) Frequency fusion to retain detail from the original
    """

    original = img.copy()  # preserved for frequency fusion

    # Adaptive parameters based on tile size
    if tile_size is not None:
        scale = tile_size / 112.0
        d_bilateral = max(3, int(9 * scale))
        clahe_grid_size = max(2, int(8 * scale))
        guided_radius = max(2, int(8 * scale))
        bilateral_fallback_d = max(3, int(7 * scale))
    else:
        d_bilateral = 9
        clahe_grid_size = 8
        guided_radius = 8
        bilateral_fallback_d = 7

    # 1) DENOISE - bilateral to keep edges
    img = cv2.bilateralFilter(
        img,
        d=d_bilateral,                # adaptive kernel size
        sigmaColor=40,
        sigmaSpace=40
    )


    # 3) Edge-preserving smoothing (guided filter if available)
    try:
        img = cv2.ximgproc.guidedFilter(
            guide=img,
            src=img,
            radius=guided_radius,
            eps=1e-2
        )
    except Exception:
        # fallback: still edge-preserving
        img = cv2.bilateralFilter(img, d=bilateral_fallback_d, sigmaColor=20, sigmaSpace=20)

    # 4) Soft unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.1)
    sharp = cv2.addWeighted(img, 1.12, blur, -0.12, 0)

    # 5) Frequency fusion with details
    final = cv2.addWeighted(original, 0.55, sharp, 0.45, 0)

    return final


# ===================================
# VISUALIZATION OF TILE PROCESSING
# ===================================

def visualize_tile_borders(tile: np.ndarray, strip_width: int = 1) -> np.ndarray:
    """
    Visualize tile borders by highlighting extraction strips.
    Useful for understanding which parts of the tile are used for matching.
    
    Args:
        tile: Input tile image (BGR)
        strip_width: Width of border strips to highlight
        
    Returns:
        Visualization showing border regions
    """
    vis = tile.copy()
    h, w = vis.shape[:2]
    sw = min(strip_width, h // 2, w // 2)
    
    # Top border (blue)
    vis[0:sw, :] = cv2.addWeighted(vis[0:sw, :], 0.5, np.zeros_like(vis[0:sw, :]), 0, 0)
    vis[0:sw, :, 0] = 255
    
    # Right border (green)
    vis[:, w-sw:w] = cv2.addWeighted(vis[:, w-sw:w], 0.5, np.zeros_like(vis[:, w-sw:w]), 0, 0)
    vis[:, w-sw:w, 1] = 255
    
    # Bottom border (red)
    vis[h-sw:h, :] = cv2.addWeighted(vis[h-sw:h, :], 0.5, np.zeros_like(vis[h-sw:h, :]), 0, 0)
    vis[h-sw:h, :, 2] = 255
    
    # Left border (yellow)
    vis[:, 0:sw] = cv2.addWeighted(vis[:, 0:sw], 0.5, np.zeros_like(vis[:, 0:sw]), 0, 0)
    vis[:, 0:sw, 1] = 255
    vis[:, 0:sw, 2] = 255
    
    return vis


def visualize_tile_processing_stages(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """
    Display before/after comparison of tile enhancement.
    
    Args:
        original: Original tile image
        enhanced: Enhanced tile image
        
    Returns:
        Side-by-side comparison image
    """
    # Resize if needed to match dimensions
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Create composite
    h, w = original.shape[:2]
    composite = np.zeros((h, w*2 + 10, 3), dtype=original.dtype)
    
    composite[:, :w] = original
    composite[:, w+10:] = enhanced
    
    return composite


def visualize_grid_overlay(source_image: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    """
    Overlay grid lines on the source image to visualize how it will be cut.
    
    Args:
        source_image: The original image to be cut
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        
    Returns:
        Image with grid overlay
    """
    vis = source_image.copy()
    h, w = vis.shape[:2]
    
    # Draw vertical lines
    tile_w = w / grid_cols
    for col in range(1, grid_cols):
        x = int(col * tile_w)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 255), 2)
    
    # Draw horizontal lines
    tile_h = h / grid_rows
    for row in range(1, grid_rows):
        y = int(row * tile_h)
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 2)
    
    return vis


def visualize_tiles_montage(tiles: List[np.ndarray], tiles_per_row: int = 5) -> np.ndarray:
    """
    Create a montage of all tiles in a grid layout for inspection.
    
    Args:
        tiles: List of tile images
        tiles_per_row: How many tiles to display per row
        
    Returns:
        Montage image
    """
    if not tiles:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    tile_h, tile_w = tiles[0].shape[:2]
    n_tiles = len(tiles)
    rows = (n_tiles + tiles_per_row - 1) // tiles_per_row
    
    margin = 5
    montage_h = rows * (tile_h + margin) + margin
    montage_w = tiles_per_row * (tile_w + margin) + margin
    
    montage = np.ones((montage_h, montage_w, 3), dtype=tiles[0].dtype) * 200
    
    for idx, tile in enumerate(tiles):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        y = margin + row * (tile_h + margin)
        x = margin + col * (tile_w + margin)
        
        # Resize if necessary
        if tile.shape[:2] != (tile_h, tile_w):
            tile = cv2.resize(tile, (tile_w, tile_h))
        
        montage[y:y+tile_h, x:x+tile_w] = tile
    
    return montage
