# phase2/features.py
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

# ----------------- Loader & sanitizer -----------------

def load_tiles_from_phase1(phase1_root: str, group: str, image: str) -> List[Dict]:
    tile_dir = os.path.join(phase1_root, group, image, "tiles")
    if not os.path.isdir(tile_dir):
        raise FileNotFoundError(tile_dir)
    files = sorted([f for f in os.listdir(tile_dir) if f.endswith(".png") and not f.endswith("_mask.png")])
    tiles = []
    for fn in files:
        base = fn[:-4]
        img = cv2.imread(os.path.join(tile_dir, fn), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask_path = os.path.join(tile_dir, f"{base}_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255
        else:
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        tiles.append({"id": base, "img": img, "mask": mask, "path": os.path.join(tile_dir, fn)})
    if len(tiles) == 0:
        raise RuntimeError("No tiles loaded")
    return tiles

# ----------------- Border strips and RMS -----------------

def extract_border_strips(tile_img: np.ndarray, tile_mask: np.ndarray, strip_frac: float):
    h,w = tile_img.shape[:2]
    ph = max(1, int(round(h * strip_frac)))
    pw = max(1, int(round(w * strip_frac)))
    # take inner strips (avoid extreme outer row/col)
    top_img = tile_img[0:ph, :]
    top_mask = tile_mask[0:ph, :]
    bottom_img = tile_img[h-ph:h, :]
    bottom_mask = tile_mask[h-ph:h, :]
    left_img = tile_img[:, 0:pw]
    left_mask = tile_mask[:, 0:pw]
    right_img = tile_img[:, w-pw:w]
    right_mask = tile_mask[:, w-pw:w]
    return top_img, top_mask, bottom_img, bottom_mask, left_img, left_mask, right_img, right_mask

def rms_distance(a_img, a_mask, b_img, b_mask) -> float:
    # Ensure compatible shapes: align by cropping to min dims if necessary
    if a_img.shape != b_img.shape:
        mh = min(a_img.shape[0], b_img.shape[0])
        mw = min(a_img.shape[1], b_img.shape[1])
        a_img = a_img[:mh, :mw]
        b_img = b_img[:mh, :mw]
        a_mask = a_mask[:mh, :mw]
        b_mask = b_mask[:mh, :mw]
    # masked difference
    mask = (a_mask > 0) & (b_mask > 0)
    if not np.any(mask):
        return 255.0
    diff = (a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    diff = diff[mask]
    if diff.size == 0:
        return 255.0
    return float(np.sqrt(np.mean(diff)))

def compute_border_distances(tiles: List[Dict], strip_frac: float = 0.1):
    n = len(tiles)
    H_dist = np.zeros((n,n), dtype=np.float32)
    V_dist = np.zeros((n,n), dtype=np.float32)
    strips = []
    for t in tiles:
        strips.append(extract_border_strips(t["img"], t["mask"], strip_frac))
    for i in range(n):
        for j in range(n):
            # right(i) vs left(j)
            _,_,_,_,_,_, right_img, right_mask = strips[i]
            top_img_j, top_mask_j, bottom_img_j, bottom_mask_j, left_img_j, left_mask_j, _, _ = strips[j]
            H_dist[i,j] = rms_distance(right_img, right_mask, left_img_j, left_mask_j)
            # bottom(i) vs top(j)
            _,_, bottom_img_i, bottom_mask_i, _,_,_,_ = strips[i]
            top_img_j, top_mask_j,_,_,_,_,_,_ = strips[j]
            V_dist[i,j] = rms_distance(bottom_img_i, bottom_mask_i, top_img_j, top_mask_j)
    return H_dist, V_dist

def distance_to_similarity_matrix(dist_mat: np.ndarray) -> np.ndarray:
    # invert with scale factor tuned empirically
    return (1.0 / (1.0 + (dist_mat / 50.0))).astype(np.float32)

# ----------------- Global similarity -----------------

def compute_hsv_hist_similarity(imgA, imgB):
    hsvA = cv2.cvtColor(imgA, cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(imgB, cv2.COLOR_BGR2HSV)
    histA = cv2.calcHist([hsvA], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histB = cv2.calcHist([hsvB], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histA = cv2.normalize(histA, None).flatten()
    histB = cv2.normalize(histB, None).flatten()
    d = cv2.compareHist(histA, histB, cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - np.clip(d, 0, 1))

def compute_ncc_similarity(imgA, imgB):
    try:
        A = cv2.resize(imgA, (64,64)).astype(np.float32)
        B = cv2.resize(imgB, (64,64)).astype(np.float32)
        A -= A.mean(); B -= B.mean()
        denom = (np.linalg.norm(A)*np.linalg.norm(B) + 1e-12)
        return float(np.clip((A*B).sum()/denom, -1, 1))
    except:
        return 0.0

def compute_orb_ratio(imgA, imgB):
    orb = cv2.ORB_create(300)
    A = cv2.resize(imgA, (128,128)); B = cv2.resize(imgB, (128,128))
    kp1, des1 = orb.detectAndCompute(A, None)
    kp2, des2 = orb.detectAndCompute(B, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except:
        return 0.0
    if len(matches) == 0:
        return 0.0
    return float(min(len(matches) / max(1, min(len(kp1), len(kp2))), 1.0))

def compute_global_similarity(tiles: List[Dict]) -> np.ndarray:
    n = len(tiles)
    G = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                G[i,j] = 1.0; continue
            imgA = tiles[i]["img"]; imgB = tiles[j]["img"]
            h = compute_hsv_hist_similarity(imgA, imgB)
            o = compute_orb_ratio(imgA, imgB)
            ncc = compute_ncc_similarity(imgA, imgB)
            G[i,j] = float(np.mean([h,o,ncc]))
    return G

# ----------------- Object segmentation & matching -----------------

def segment_objects_in_tile(mask: np.ndarray, min_area: int = 30):
    if mask is None:
        return []
    labels_mask = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(labels_mask, connectivity=8)
    objs = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT]); y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH]); h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        submask = (labels[y:y+h, x:x+w] == lbl).astype(np.uint8) * 255
        cx, cy = centroids[lbl]
        objs.append({"bbox": (x,y,w,h), "mask": submask, "centroid": (float(cx), float(cy)), "area": area})
    return objs

def compute_object_descriptors_for_tiles(tiles: List[Dict]):
    return [ segment_objects_in_tile(t["mask"], min_area=20) for t in tiles ]

def _hu_moments_similarity(maskA: np.ndarray, maskB: np.ndarray) -> float:
    def hu_vals(m):
        m = (m>0).astype(np.uint8)
        mom = cv2.moments(m)
        hu = cv2.HuMoments(mom).flatten()
        for i in range(len(hu)):
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)
        return hu
    try:
        a = hu_vals(maskA); b = hu_vals(maskB)
        d = np.linalg.norm(a-b, ord=1)
        return float(1.0/(1.0 + d))
    except:
        return 0.0

def _orb_object_match_ratio(patchA: np.ndarray, maskA: np.ndarray, patchB: np.ndarray, maskB: np.ndarray) -> float:
    # sanitize masks
    def sanitize(patch, mask):
        if mask is None:
            m = np.ones((patch.shape[0], patch.shape[1]), dtype=np.uint8)*255
        else:
            m = mask.copy()
            if m.ndim==3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            if m.shape[:2] != patch.shape[:2]:
                m = cv2.resize(m, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)*255
        return m
    try:
        if patchA is None or patchB is None:
            return 0.0
        mA = sanitize(patchA, maskA); mB = sanitize(patchB, maskB)
        pA = cv2.bitwise_and(patchA, patchA, mask=mA)
        pB = cv2.bitwise_and(patchB, patchB, mask=mB)
        pA = cv2.resize(pA, (128,128), interpolation=cv2.INTER_AREA)
        pB = cv2.resize(pB, (128,128), interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create(300)
        kp1, des1 = orb.detectAndCompute(pA, None)
        kp2, des2 = orb.detectAndCompute(pB, None)
        if des1 is None or des2 is None:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0.0
        ratio = len(matches) / max(1, min(len(kp1), len(kp2)))
        return float(min(ratio, 1.0))
    except:
        return 0.0

def compute_object_similarity_matrix(tiles: List[Dict], objs_per_tile: List[List[Dict]], direction: str = "horizontal", edge_zone_frac: float = 0.25, alpha_obj_shape: float = 0.6) -> np.ndarray:
    n = len(tiles); out = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j:
                out[i,j] = 1.0; continue
            objsA = objs_per_tile[i]; objsB = objs_per_tile[j]
            if not objsA or not objsB:
                out[i,j] = 0.0; continue
            best = 0.0
            for a in objsA:
                for b in objsB:
                    xa,ya,wa,ha = a["bbox"]; xb,yb,wb,hb = b["bbox"]
                    patchA = tiles[i]["img"][ya:ya+ha, xa:xa+wa]
                    patchB = tiles[j]["img"][yb:yb+hb, xb:xb+wb]
                    hu = _hu_moments_similarity(a["mask"], b["mask"])
                    orb = _orb_object_match_ratio(patchA, a["mask"], patchB, b["mask"])
                    score = alpha_obj_shape * hu + (1.0-alpha_obj_shape) * orb
                    if score > best: best = score
            out[i,j] = float(best)
    return out

# ----------------- Edge-based similarity (Canny IoU) -----------------

def _edge_map(img: np.ndarray, low=50, high=150) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, low, high)
    return (edges>0).astype(np.uint8)*255

def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    A = (a>0).astype(np.uint8); B = (b>0).astype(np.uint8)
    inter = int((A & B).sum()); union = int((A | B).sum())
    if union == 0: return 0.0
    return float(inter/union)

def compute_edge_similarity(tiles: List[Dict], strip_frac: float = 0.12, canny_low=50, canny_high=150) -> Tuple[np.ndarray, np.ndarray]:
    n = len(tiles)
    edges = [ _edge_map(t["img"], low=canny_low, high=canny_high) for t in tiles ]
    def extract(em, dir):
        h,w = em.shape[:2]
        sh = max(1, int(round(h*strip_frac)))
        sw = max(1, int(round(w*strip_frac)))
        if dir=="top": return em[0:sh, sw:w-sw]
        if dir=="bottom": return em[h-sh:h, sw:w-sw]
        if dir=="left": return em[sh:h-sh, 0:sw]
        if dir=="right": return em[sh:h-sh, w-sw:w]
        return em
    strips = []
    for em in edges:
        strips.append({"top": extract(em,"top"), "bottom": extract(em,"bottom"), "left": extract(em,"left"), "right": extract(em,"right")})
    H = np.zeros((n,n), dtype=np.float32); V = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            r = strips[i]["right"]; l = strips[j]["left"]
            mh = min(r.shape[0], l.shape[0]); mw = min(r.shape[1], l.shape[1])
            if mh<=0 or mw<=0: H[i,j] = 0.0
            else: H[i,j] = _binary_iou(r[:mh,:mw], l[:mh,:mw])
            b = strips[i]["bottom"]; t = strips[j]["top"]
            mh = min(b.shape[0], t.shape[0]); mw = min(b.shape[1], t.shape[1])
            if mh<=0 or mw<=0: V[i,j] = 0.0
            else: V[i,j] = _binary_iou(b[:mh,:mw], t[:mh,:mw])
    return H, V
