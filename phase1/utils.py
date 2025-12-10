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


def adaptive_thresh(gray, blockSize=51, C=2):
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


def morphological_clean(mask, kernel_size=5, min_area=500):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def edge_map(gray, low=50, high=150):
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

def smart_enhance(img: np.ndarray) -> np.ndarray:
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

    # 1) DENOISE – bilateral to keep edges
    img = cv2.bilateralFilter(
        img,
        d=9,                # small kernel, avoids watercolor effect
        sigmaColor=40,
        sigmaSpace=40
    )

    # 2) CLAHE on luminance only
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    # 3) Edge-preserving smoothing (guided filter if available)
    try:
        img = cv2.ximgproc.guidedFilter(
            guide=img,
            src=img,
            radius=8,
            eps=1e-2
        )
    except Exception:
        # fallback: still edge-preserving
        img = cv2.bilateralFilter(img, d=7, sigmaColor=20, sigmaSpace=20)

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
                 morph_min_area: int = 300) -> np.ndarray:
    """
    Segment puzzle piece from a tile image.
    Uses Otsu, then morphology, with adaptive fallback.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

    mask = otsu_thresh(gray)
    mask = morphological_clean(mask, kernel_size=morph_kernel, min_area=morph_min_area)

    # fallback: if almost nothing detected, try adaptive
    if np.sum(mask) < 0.01 * mask.size:
        mask = adaptive_thresh(gray)
        mask = morphological_clean(mask, kernel_size=morph_kernel, min_area=morph_min_area)

    return mask


def isolate_largest_component(mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Keep only the largest connected component (the main puzzle piece)
    """
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
