# phase2/features.py
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

# ----------------- Loader & sanitizer -----------------

def load_tiles_from_phase1(source_root: str, category: str, identifier: str) -> List[Dict]:
    resource_dir = os.path.join(source_root, category, identifier, "tiles")
    if not os.path.isdir(resource_dir):
        raise FileNotFoundError(resource_dir)
    filenames = sorted([element for element in os.listdir(resource_dir) if element.endswith(".png") and not element.endswith("_mask.png")])
    collection = []
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

def extract_border_strips(source_img: np.ndarray, source_mask: np.ndarray, fraction: float):
    height, width = source_img.shape[:2]
    pixel_height = max(1, int(round(height * fraction)))
    pixel_width = max(1, int(round(width * fraction)))
    # take inner strips (avoid extreme outer row/col)
    upper_img = source_img[0:pixel_height, :]
    upper_mask = source_mask[0:pixel_height, :]
    lower_img = source_img[height-pixel_height:height, :]
    lower_mask = source_mask[height-pixel_height:height, :]
    start_img = source_img[:, 0:pixel_width]
    start_mask = source_mask[:, 0:pixel_width]
    end_img = source_img[:, width-pixel_width:width]
    end_mask = source_mask[:, width-pixel_width:width]
    return upper_img, upper_mask, lower_img, lower_mask, start_img, start_mask, end_img, end_mask

def rms_distance(first_img, first_mask, second_img, second_mask) -> float:
    # Ensure compatible shapes: align by cropping to min dims if necessary
    if first_img.shape != second_img.shape:
        min_h = min(first_img.shape[0], second_img.shape[0])
        min_w = min(first_img.shape[1], second_img.shape[1])
        first_img = first_img[:min_h, :min_w]
        second_img = second_img[:min_h, :min_w]
        first_mask = first_mask[:min_h, :min_w]
        second_mask = second_mask[:min_h, :min_w]
    # masked difference
    valid_mask = (first_mask > 0) & (second_mask > 0)
    if not np.any(valid_mask):
        return 255.0
    squared_diff = (first_img.astype(np.float32) - second_img.astype(np.float32)) ** 2
    if squared_diff.ndim == 3:
        squared_diff = squared_diff.mean(axis=2)
    squared_diff = squared_diff[valid_mask]
    if squared_diff.size == 0:
        return 255.0
    return float(np.sqrt(np.mean(squared_diff)))

def compute_border_distances(tiles: List[Dict], fraction: float = 0.1):
    count = len(tiles)
    horizontal_dist = np.zeros((count,count), dtype=np.float32)
    vertical_dist = np.zeros((count,count), dtype=np.float32)
    extracted_strips = []
    for tile in tiles:
        extracted_strips.append(extract_border_strips(tile["img"], tile["mask"], fraction))
    for idx_i in range(count):
        for idx_j in range(count):
            # right(idx_i) vs left(idx_j)
            _,_,_,_,_,_, right_image, right_mask = extracted_strips[idx_i]
            upper_img_j, upper_mask_j, lower_img_j, lower_mask_j, left_image_j, left_mask_j, _, _ = extracted_strips[idx_j]
            horizontal_dist[idx_i,idx_j] = rms_distance(right_image, right_mask, left_image_j, left_mask_j)
            # bottom(idx_i) vs top(idx_j)
            _,_, lower_image_i, lower_mask_i, _,_,_,_ = extracted_strips[idx_i]
            upper_img_j, upper_mask_j,_,_,_,_,_,_ = extracted_strips[idx_j]
            vertical_dist[idx_i,idx_j] = rms_distance(lower_image_i, lower_mask_i, upper_img_j, upper_mask_j)
    return horizontal_dist, vertical_dist

def distance_to_similarity_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # invert with scale factor tuned empirically
    return (1.0 / (1.0 + (distance_matrix / 50.0))).astype(np.float32)

# ----------------- Global similarity -----------------

def compute_hsv_hist_similarity(image_a, image_b):
    hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
    histogram_a = cv2.calcHist([hsv_a], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histogram_b = cv2.calcHist([hsv_b], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    histogram_a = cv2.normalize(histogram_a, None).flatten()
    histogram_b = cv2.normalize(histogram_b, None).flatten()
    distance = cv2.compareHist(histogram_a, histogram_b, cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - np.clip(distance, 0, 1))

def compute_ncc_similarity(image_a, image_b):
    try:
        matrix_a = cv2.resize(image_a, (64,64)).astype(np.float32)
        matrix_b = cv2.resize(image_b, (64,64)).astype(np.float32)
        matrix_a -= matrix_a.mean(); matrix_b -= matrix_b.mean()
        denominator = (np.linalg.norm(matrix_a)*np.linalg.norm(matrix_b) + 1e-12)
        return float(np.clip((matrix_a*matrix_b).sum()/denominator, -1, 1))
    except:
        return 0.0

def compute_orb_ratio(image_a, image_b):
    orb_detector = cv2.ORB_create(300)
    img_a_resized = cv2.resize(image_a, (128,128)); img_b_resized = cv2.resize(image_b, (128,128))
    keypoints_a, descriptors_a = orb_detector.detectAndCompute(img_a_resized, None)
    keypoints_b, descriptors_b = orb_detector.detectAndCompute(img_b_resized, None)
    if descriptors_a is None or descriptors_b is None:
        return 0.0
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matched_pairs = brute_force.match(descriptors_a, descriptors_b)
    except:
        return 0.0
    if len(matched_pairs) == 0:
        return 0.0
    return float(min(len(matched_pairs) / max(1, min(len(keypoints_a), len(keypoints_b))), 1.0))

def compute_global_similarity(tiles: List[Dict]) -> np.ndarray:
    tile_count = len(tiles)
    similarity_matrix = np.zeros((tile_count,tile_count), dtype=np.float32)
    for row_idx in range(tile_count):
        for col_idx in range(tile_count):
            if row_idx == col_idx:
                similarity_matrix[row_idx,col_idx] = 1.0; continue
            image_a = tiles[row_idx]["img"]; image_b = tiles[col_idx]["img"]
            hsv_sim = compute_hsv_hist_similarity(image_a, image_b)
            orb_sim = compute_orb_ratio(image_a, image_b)
            ncc_sim = compute_ncc_similarity(image_a, image_b)
            similarity_matrix[row_idx,col_idx] = float(np.mean([hsv_sim,orb_sim,ncc_sim]))
    return similarity_matrix

# ----------------- Object segmentation & matching -----------------

def segment_objects_in_tile(mask_input: np.ndarray, min_area: int = 30):
    if mask_input is None:
        return []
    binary_mask = (mask_input > 0).astype(np.uint8)
    component_count, label_map, component_stats, component_centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    detected_objects = []
    for label_idx in range(1, component_count):
        object_area = int(component_stats[label_idx, cv2.CC_STAT_AREA])
        if object_area < min_area:
            continue
        pos_x = int(component_stats[label_idx, cv2.CC_STAT_LEFT]); pos_y = int(component_stats[label_idx, cv2.CC_STAT_TOP])
        size_w = int(component_stats[label_idx, cv2.CC_STAT_WIDTH]); size_h = int(component_stats[label_idx, cv2.CC_STAT_HEIGHT])
        object_mask = (label_map[pos_y:pos_y+size_h, pos_x:pos_x+size_w] == label_idx).astype(np.uint8) * 255
        center_x, center_y = component_centroids[label_idx]
        detected_objects.append({"bbox": (pos_x,pos_y,size_w,size_h), "mask": object_mask, "centroid": (float(center_x), float(center_y)), "area": object_area})
    return detected_objects

def compute_object_descriptors_for_tiles(tiles: List[Dict]):
    return [ segment_objects_in_tile(tile_item["mask"], min_area=20) for tile_item in tiles ]

def _hu_moments_similarity(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    def compute_hu_values(input_mask):
        input_mask = (input_mask>0).astype(np.uint8)
        moments = cv2.moments(input_mask)
        hu_moments = cv2.HuMoments(moments).flatten()
        for idx in range(len(hu_moments)):
            hu_moments[idx] = -np.sign(hu_moments[idx]) * np.log10(abs(hu_moments[idx]) + 1e-12)
        return hu_moments
    try:
        values_a = compute_hu_values(mask_1); values_b = compute_hu_values(mask_2)
        norm_distance = np.linalg.norm(values_a-values_b, ord=1)
        return float(1.0/(1.0 + norm_distance))
    except:
        return 0.0

def _orb_object_match_ratio(patch_1: np.ndarray, mask_1: np.ndarray, patch_2: np.ndarray, mask_2: np.ndarray) -> float:
    # sanitize masks
    def sanitize_mask(patch_data, mask_data):
        if mask_data is None:
            output_mask = np.ones((patch_data.shape[0], patch_data.shape[1]), dtype=np.uint8)*255
        else:
            output_mask = mask_data.copy()
            if output_mask.ndim==3:
                output_mask = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY)
            if output_mask.shape[:2] != patch_data.shape[:2]:
                output_mask = cv2.resize(output_mask, (patch_data.shape[1], patch_data.shape[0]), interpolation=cv2.INTER_NEAREST)
            output_mask = (output_mask > 0).astype(np.uint8)*255
        return output_mask
    try:
        if patch_1 is None or patch_2 is None:
            return 0.0
        cleaned_mask_1 = sanitize_mask(patch_1, mask_1); cleaned_mask_2 = sanitize_mask(patch_2, mask_2)
        masked_patch_1 = cv2.bitwise_and(patch_1, patch_1, mask=cleaned_mask_1)
        masked_patch_2 = cv2.bitwise_and(patch_2, patch_2, mask=cleaned_mask_2)
        masked_patch_1 = cv2.resize(masked_patch_1, (128,128), interpolation=cv2.INTER_AREA)
        masked_patch_2 = cv2.resize(masked_patch_2, (128,128), interpolation=cv2.INTER_AREA)
        orb_detector = cv2.ORB_create(300)
        kp_set_1, desc_set_1 = orb_detector.detectAndCompute(masked_patch_1, None)
        kp_set_2, desc_set_2 = orb_detector.detectAndCompute(masked_patch_2, None)
        if desc_set_1 is None or desc_set_2 is None:
            return 0.0
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matched_points = brute_force_matcher.match(desc_set_1, desc_set_2)
        if not matched_points:
            return 0.0
        match_ratio = len(matched_points) / max(1, min(len(kp_set_1), len(kp_set_2)))
        return float(min(match_ratio, 1.0))
    except:
        return 0.0

def compute_object_similarity_matrix(tiles: List[Dict], objs_per_tile: List[List[Dict]], direction: str = "horizontal", edge_zone_frac: float = 0.25, shape_weight: float = 0.6) -> np.ndarray:
    tile_count = len(tiles); output_matrix = np.zeros((tile_count,tile_count), dtype=np.float32)
    for tile_i in range(tile_count):
        for tile_j in range(tile_count):
            if tile_i==tile_j:
                output_matrix[tile_i,tile_j] = 1.0; continue
            objects_i = objs_per_tile[tile_i]; objects_j = objs_per_tile[tile_j]
            if not objects_i or not objects_j:
                output_matrix[tile_i,tile_j] = 0.0; continue
            best_score = 0.0
            for obj_a in objects_i:
                for obj_b in objects_j:
                    bbox_x_a, bbox_y_a, bbox_w_a, bbox_h_a = obj_a["bbox"]
                    bbox_x_b, bbox_y_b, bbox_w_b, bbox_h_b = obj_b["bbox"]
                    extracted_patch_a = tiles[tile_i]["img"][bbox_y_a:bbox_y_a+bbox_h_a, bbox_x_a:bbox_x_a+bbox_w_a]
                    extracted_patch_b = tiles[tile_j]["img"][bbox_y_b:bbox_y_b+bbox_h_b, bbox_x_b:bbox_x_b+bbox_w_b]
                    hu_similarity = _hu_moments_similarity(obj_a["mask"], obj_b["mask"])
                    orb_similarity = _orb_object_match_ratio(extracted_patch_a, obj_a["mask"], extracted_patch_b, obj_b["mask"])
                    combined_score = shape_weight * hu_similarity + (1.0-shape_weight) * orb_similarity
                    if combined_score > best_score: best_score = combined_score
            output_matrix[tile_i,tile_j] = float(best_score)
    return output_matrix

# ----------------- Edge-based similarity (Canny IoU) -----------------

def _edge_map(image_input: np.ndarray, low=50, high=150) -> np.ndarray:
    grayscale = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim==3 else image_input
    smoothed = cv2.GaussianBlur(grayscale, (3,3), 0)
    edge_output = cv2.Canny(smoothed, low, high)
    return (edge_output>0).astype(np.uint8)*255

def _binary_iou(array_a: np.ndarray, array_b: np.ndarray) -> float:
    binary_a = (array_a>0).astype(np.uint8); binary_b = (array_b>0).astype(np.uint8)
    intersection = int((binary_a & binary_b).sum()); combined_union = int((binary_a | binary_b).sum())
    if combined_union == 0: return 0.0
    return float(intersection/combined_union)

def compute_edge_similarity(tiles: List[Dict], fraction: float = 0.12, canny_low=50, canny_high=150) -> Tuple[np.ndarray, np.ndarray]:
    tile_count = len(tiles)
    edge_maps = [ _edge_map(tile_data["img"], low=canny_low, high=canny_high) for tile_data in tiles ]
    def extract_region(edge_map, region_dir):
        map_h, map_w = edge_map.shape[:2]
        strip_h = max(1, int(round(map_h*fraction)))
        strip_w = max(1, int(round(map_w*fraction)))
        if region_dir=="top": return edge_map[0:strip_h, strip_w:map_w-strip_w]
        if region_dir=="bottom": return edge_map[map_h-strip_h:map_h, strip_w:map_w-strip_w]
        if region_dir=="left": return edge_map[strip_h:map_h-strip_h, 0:strip_w]
        if region_dir=="right": return edge_map[strip_h:map_h-strip_h, map_w-strip_w:map_w]
        return edge_map
    extracted_strips = []
    for edge_map_item in edge_maps:
        extracted_strips.append({"top": extract_region(edge_map_item,"top"), "bottom": extract_region(edge_map_item,"bottom"), "left": extract_region(edge_map_item,"left"), "right": extract_region(edge_map_item,"right")})
    horizontal_matrix = np.zeros((tile_count,tile_count), dtype=np.float32); vertical_matrix = np.zeros((tile_count,tile_count), dtype=np.float32)
    for idx_i in range(tile_count):
        for idx_j in range(tile_count):
            right_region = extracted_strips[idx_i]["right"]; left_region = extracted_strips[idx_j]["left"]
            common_h = min(right_region.shape[0], left_region.shape[0]); common_w = min(right_region.shape[1], left_region.shape[1])
            if common_h<=0 or common_w<=0: horizontal_matrix[idx_i,idx_j] = 0.0
            else: horizontal_matrix[idx_i,idx_j] = _binary_iou(right_region[:common_h,:common_w], left_region[:common_h,:common_w])
            bottom_region = extracted_strips[idx_i]["bottom"]; top_region = extracted_strips[idx_j]["top"]
            common_h = min(bottom_region.shape[0], top_region.shape[0]); common_w = min(bottom_region.shape[1], top_region.shape[1])
            if common_h<=0 or common_w<=0: vertical_matrix[idx_i,idx_j] = 0.0
            else: vertical_matrix[idx_i,idx_j] = _binary_iou(bottom_region[:common_h,:common_w], top_region[:common_h,:common_w])
    return horizontal_matrix, vertical_matrix
