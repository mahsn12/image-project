import cv2
import numpy as np


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
