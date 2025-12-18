import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


def smart_enhance(img: np.ndarray, tile_size: int = None) -> np.ndarray:
    # Enhance image: denoise, smooth, sharpen, and retain detail
    original = img.copy()

    # Adapt parameters based on tile size
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

    # Denoise with bilateral filter to preserve edges
    img = cv2.bilateralFilter(
        img,
        d=d_bilateral,
        sigmaColor=40,
        sigmaSpace=40
    )

    # Smooth edges while preserving contours
    try:
        img = cv2.ximgproc.guidedFilter(
            guide=img,
            src=img,
            radius=guided_radius,
            eps=1e-2
        )
    except Exception:
        img = cv2.bilateralFilter(img, d=bilateral_fallback_d, sigmaColor=20, sigmaSpace=20)

    # Sharpen image with unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.1)
    sharp = cv2.addWeighted(img, 1.12, blur, -0.12, 0)

    # Blend original and sharpened versions
    final = cv2.addWeighted(original, 0.55, sharp, 0.45, 0)

    return final
