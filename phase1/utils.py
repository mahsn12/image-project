import cv2
import numpy as np

def apply_clahe(img_bgr, clipLimit=2.0, tileGridSize=(8,8)):
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
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
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
        cv2.drawContours(out, [c], -1, (0,255,0), 2)
    return out
