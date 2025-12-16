import cv2
import numpy as np


# ==========================
 # BASIC HELPERS
# ==========================

def apply_clahe(img_bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def bilateral_smooth(img_bgr, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img_bgr, d, sigmaColor, sigmaSpace)


def unsharp_mask(img_bgr, sigma=3, strength=1.5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    sharp = cv2.addWeighted(gray, strength, blur, 1 - strength, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def denoise(img_bgr, h=10):
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)


def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def adaptive_thresh(gray, blockSize=51, C=2, tile_size=None):
    """Adaptive threshold with size-scaled blockSize."""
    if tile_size is not None:
        # Scale blockSize: 112px -> 51, 56px -> 25, 28px -> 13
        scale = tile_size / 112.0
        blockSize = max(3, int(blockSize * scale))
    if blockSize % 2 == 0:
        blockSize += 1
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize, C
    )


def otsu_thresh(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def morphological_clean(mask, kernel_size=5, min_area=500, tile_size=None):
    """Morphological cleaning with adaptive parameters based on tile size."""
    if tile_size is not None:
        # Scale parameters: 112px -> 1.0x, 56px -> 0.5x, 28px -> 0.25x
        scale = tile_size / 112.0
        kernel_size = max(3, int(kernel_size * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        min_area = max(20, int(min_area * scale * scale))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def edge_map(gray, low=50, high=150, tile_size=None):
    """Edge detection with adaptive thresholds based on tile size."""
    if tile_size is not None:
        # Scale thresholds: 112px -> 1.0x, 56px -> 0.5x, 28px -> 0.25x
        scale = tile_size / 112.0
        low = max(10, int(low * scale))
        high = max(30, int(high * scale))
    return cv2.Canny(gray, low, high)


def extract_contours_from_mask(mask):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(cnts, key=cv2.contourArea, reverse=True)


def draw_contours_on_image(img_bgr, contours, max_cnt=50):
    out = img_bgr.copy()
    for i, c in enumerate(contours[:max_cnt]):
        cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
    return out


# ==========================
# SMART ENHANCEMENT PIPELINE
# ==========================

def smart_enhance(img: np.ndarray, tile_size: int = None) -> np.ndarray:
    """
    DENOISE FIRST then enhance:
      1. Bilateral denoise (edge-preserving)
      2. CLAHE on L-channel (local contrast, no color damage)
      3. Guided filter (or small bilateral fallback)
      4. Soft unsharp mask (crisper edges, no halos)
    5. Frequency fusion with details

    Returns: enhanced image with sharper edges & preserved details.
    """

    original = img.copy()  # for frequency fusion
    
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

    # 1) DENOISE – bilateral to keep edges
    img = cv2.bilateralFilter(
        img,
        d=d_bilateral,                # adaptive kernel size
        sigmaColor=40,
        sigmaSpace=40
    )

    # 2) CLAHE on luminance only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(clahe_grid_size, clahe_grid_size))
    l2 = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

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


# ==========================
# TILE SEGMENTATION HELPERS
# ==========================

def segment_tile(tile_bgr: np.ndarray,
                 morph_kernel: int = 5,
                 morph_min_area: int = 300,
                 tile_size: int = None) -> np.ndarray:
    """
    Segment puzzle piece from a tile image.
    Uses Otsu, then morphology, with adaptive fallback.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

    mask = otsu_thresh(gray)
    mask = morphological_clean(mask, kernel_size=morph_kernel, min_area=morph_min_area, tile_size=tile_size)

    # fallback: if almost nothing detected, try adaptive
    if np.sum(mask) < 0.01 * mask.size:
        mask = adaptive_thresh(gray, tile_size=tile_size)
        mask = morphological_clean(mask, kernel_size=morph_kernel, min_area=morph_min_area, tile_size=tile_size)

    return mask


def isolate_largest_component(mask: np.ndarray, min_area: int = 300, tile_size: int = None) -> np.ndarray:
    """
    Keep only the largest connected component (the main puzzle piece)
    """
    if tile_size is not None:
        scale = tile_size / 112.0
        min_area = max(20, int(min_area * scale * scale))
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = np.argmax(areas) + 1

    out = np.zeros_like(mask)
    if areas[idx - 1] >= min_area:
        out[labels == idx] = 255
    return out


def tile_on_white(tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask so tile shows the piece on a white background.
    """
    clean = np.full(tile_bgr.shape, 255, dtype=np.uint8)
    clean[mask == 255] = tile_bgr[mask == 255]
    return clean


def fill_mask_holes(mask: np.ndarray, min_hole_size: int = 100, tile_size: int = None) -> np.ndarray:
    """
    Fill small holes inside the main mask component.
    
    Uses morphological closing to eliminate small voids without changing
    the outer boundary significantly.
    """
    if tile_size is not None:
        scale = tile_size / 112.0
        kernel_size = max(3, int(5 * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        min_hole_size = max(10, int(min_hole_size * scale * scale))
    else:
        kernel_size = 5
    
    # Morphological close: dilate then erode to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find and fill remaining holes using flood fill
    filled = closed.copy()
    h, w = filled.shape
    
    # Try to flood-fill from corners (background should be black/0)
    for y, x in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        if filled[y, x] == 0:
            cv2.floodFill(filled, None, (x, y), 255)
    
    # Invert to identify holes, then filter by size
    holes = 255 - filled
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    
    result = filled.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_hole_size:
            result[labels == i] = 255
    
    return result


def remove_small_specks(mask: np.ndarray, min_area: int = 50, tile_size: int = None) -> np.ndarray:
    """
    Remove isolated white specks (noise) outside the main component.
    
    Keeps only connected components above min_area threshold.
    """
    if tile_size is not None:
        scale = tile_size / 112.0
        min_area = max(5, int(min_area * scale * scale))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    
    return result


def refine_mask_edges(mask: np.ndarray, kernel_size: int = 3, tile_size: int = None) -> np.ndarray:
    """
    Smooth and slightly erode mask edges to reduce artifacts.
    
    Prevents hard boundaries from introducing errors in Phase 2 edge matching.
    """
    if tile_size is not None:
        scale = tile_size / 112.0
        kernel_size = max(2, int(kernel_size * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Slight erosion to remove thin noise on boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    
    # Dilate back slightly to restore size
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    return dilated


def enhance_mask_with_lab(tile_bgr: np.ndarray, mask_binary: np.ndarray, tile_size: int = None) -> np.ndarray:
    """
    Refine binary mask using LAB-based adaptive thresholding.
    
    Helps distinguish puzzle piece from background even with lighting variations.
    Returns improved binary mask (0 or 255).
    """
    if tile_bgr.ndim != 3 or tile_bgr.shape[2] != 3:
        return mask_binary
    
    # Scale parameters based on tile size
    if tile_size is not None:
        scale = tile_size / 112.0
        clahe_grid_size = max(2, int(8 * scale))
        blockSize = max(3, int(51 * scale))
        if blockSize % 2 == 0:
            blockSize += 1
    else:
        clahe_grid_size = 8
        blockSize = 51
    
    # Convert to LAB
    lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use L channel for refinement (luminance is most reliable)
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(clahe_grid_size, clahe_grid_size))
    l_eq = clahe.apply(l)
    
    # Adaptive threshold on enhanced L channel
    refined = cv2.adaptiveThreshold(
        l_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=blockSize, C=2
    )
    
    # Combine with original binary mask: take intersection (AND)
    # This keeps only confident piece pixels
    combined = cv2.bitwise_and(mask_binary, refined)
    
    return combined
