"""
Unified Puzzle Solver using Contour-Based Edge Matching
Combines shape analysis with constraint satisfaction to solve any puzzle size.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import time
from collections import defaultdict


class EdgeMatcher:
    """Edge matching using contour shape descriptors."""
    
    def __init__(self, tiles: List[Dict]):
        self.tiles = tiles
        self.n = len(tiles)
        self.edge_features = {}  # (tile_idx, direction) -> features (currently unused)
        # Contour-based features are disabled to avoid bad edge splits; pixel match drives scoring
        # Matching config: can be tuned
        self.strip_width = 14
        self.min_overlap_pixels = 6
        self.mask_dilate_px = 2
        # scoring weights (sum should be close to 1.0)
        self.w_border = 0.35
        self.w_masked_color = 0.25
        self.w_hist = 0.15
        self.w_grad = 0.15
        self.w_ncc = 0.10
        # self._extract_all_edges()
    
    def _extract_all_edges(self):
        """Extract edge features for all tiles, with improved contour extraction and debug output."""
        for idx, tile in enumerate(self.tiles):
            mask = tile['mask']
            img = tile['img']
            # --- Robust mask cleaning ---
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
            # --- Find contours ---
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [c for c in contours if cv2.contourArea(c) > 0.01 * mask.shape[0] * mask.shape[1]]
            if not contours:
                continue
            # --- Use the largest contour ---
            contour = max(contours, key=cv2.contourArea)
            # --- Smooth contour (optional) ---
            if len(contour) > 20:
                contour = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)
            if len(contour) < 10:
                continue
            # --- Split contour into 4 edges (improved) ---
            edges, corners = self._split_contour_to_edges_robust(contour, mask.shape)
            if not edges:
                edges = self._split_contour_to_edges(contour, mask.shape)
                corners = []
            # --- Debug: save contour and edge splits ---
            # Optional debug output
            # debug_img = img.copy()
            # cv2.drawContours(debug_img, [contour], -1, (0,255,0), 2)
            # for i, pt in enumerate(corners):
            #     cv2.circle(debug_img, tuple(int(x) for x in pt), 6, (0,0,255), -1)
            # debug_path = f"tile_{idx:02d}_contour_debug.png"
            # cv2.imwrite(debug_path, debug_img)
            # --- Extract features for each edge ---
            for direction in ['top', 'bottom', 'left', 'right']:
                if direction in edges and len(edges[direction]) > 5:
                    features = self._compute_edge_features(edges[direction], img, mask, direction)
                    self.edge_features[(idx, direction)] = features

    def _split_contour_to_edges_robust(self, contour, shape):
        """Split contour into 4 edges using corner detection (improved robustness)."""
        contour = contour.reshape(-1, 2)
        # Use Harris or Shi-Tomasi corner detection on the mask to find 4 corners
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=10)
        if corners is not None and len(corners) == 4:
            corners = np.array([c[0] for c in corners], dtype=np.int32)
        else:
            # Fallback: use bounding box corners
            x_min, y_min = contour.min(axis=0)
            x_max, y_max = contour.max(axis=0)
            corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
        # Sort corners in order (clockwise)
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:,1] - center[1], corners[:,0] - center[0])
        sort_idx = np.argsort(angles)
        corners = corners[sort_idx]
        # Split contour into 4 edges by finding closest points to each corner
        idxs = []
        for corner in corners:
            dists = np.linalg.norm(contour - corner, axis=1)
            idxs.append(np.argmin(dists))
        idxs = sorted(idxs)
        edges = {}
        edge_names = ['top', 'right', 'bottom', 'left']
        for i in range(4):
            start = idxs[i]
            end = idxs[(i+1)%4]
            if start < end:
                edge = contour[start:end+1]
            else:
                edge = np.concatenate([contour[start:], contour[:end+1]], axis=0)
            edges[edge_names[i]] = edge
        return edges, corners
    
    def _split_contour_to_edges(self, contour, shape):
        """Split contour into 4 directional edges."""
        contour = contour.reshape(-1, 2)
        h, w = shape
        
        # Get bounds
        x_min, y_min = contour.min(axis=0)
        x_max, y_max = contour.max(axis=0)
        
        # Margins for edge classification
        margin = 0.15  # 15% margin
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        
        edges = {'top': [], 'bottom': [], 'left': [], 'right': []}
        
        for pt in contour:
            x, y = pt
            
            # Classify based on position
            if y - y_min < y_margin:
                edges['top'].append(pt)
            elif y_max - y < y_margin:
                edges['bottom'].append(pt)
            elif x - x_min < x_margin:
                edges['left'].append(pt)
            elif x_max - x < x_margin:
                edges['right'].append(pt)
        
        # Convert to arrays
        for key in edges:
            if edges[key]:
                edges[key] = np.array(edges[key], dtype=np.float32)
        
        return edges
    
    def _compute_edge_features(self, edge_points, img, mask, direction):
        """Compute comprehensive edge features."""
        features = {}
        
        # 1. Fourier descriptors
        if len(edge_points) >= 10:
            complex_contour = edge_points[:, 0] + 1j * edge_points[:, 1]
            fft = np.fft.fft(complex_contour)
            descriptors = np.abs(fft)
            if descriptors[0] > 0:
                descriptors = descriptors / descriptors[0]
            features['fourier'] = descriptors[:min(20, len(descriptors))]
        else:
            features['fourier'] = np.zeros(20)
        
        # 2. Curvature
        features['curvature'] = self._compute_curvature(edge_points)
        
        # 3. Arc length
        features['arc_length'] = cv2.arcLength(edge_points.reshape(-1, 1, 2), False)
        
        # 4. Edge signature (sampled points)
        features['signature'] = self._create_edge_signature(edge_points, n_samples=64)
        
        # 5. Color gradient along edge
        features['color_gradient'] = self._edge_color_gradient(edge_points, img, mask)
        
        # 6. Edge type (straight, convex, concave)
        features['edge_type'] = self._classify_edge_type(edge_points)
        
        return features

    # ----- New helpers for robust pixel matching -----
    def _masked_histogram_similarity(self, img1, mask1, img2, mask2, bins=16):
        """Compute histogram intersection similarity between two color patches using their masks."""
        # Flatten masked pixels by channel
        if img1 is None or img2 is None:
            return 0.0
        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        masks = (mask1 > 0) & (mask2 > 0)
        if not np.any(masks):
            # Use unmasked hist if no overlap
            masks = np.ones_like(mask1, dtype=bool)
        s = 0.0
        for ch in range(3):
            h1, _ = np.histogram(img1[:,:,ch][masks], bins=bins, range=(0,255), density=True)
            h2, _ = np.histogram(img2[:,:,ch][masks], bins=bins, range=(0,255), density=True)
            # Histogram intersection
            inter = np.minimum(h1, h2).sum()
            s += inter
        return float(s / 3.0)

    def _ncc_1d_border(self, border1, border2):
        """Normalized cross-correlation for flattened border pixels (grayscale or color averaged)."""
        if border1 is None or border2 is None:
            return 0.0
        b1 = border1.astype(np.float32).ravel()
        b2 = border2.astype(np.float32).ravel()
        # Average across channels if needed
        if np.ndim(b1) > 1:
            b1 = b1.reshape(-1)
            b2 = b2.reshape(-1)
        if len(b1) != len(b2):
            L = min(len(b1), len(b2))
            b1 = b1[:L]
            b2 = b2[:L]
        b1 = b1 - b1.mean()
        b2 = b2 - b2.mean()
        denom = (np.std(b1) * np.std(b2))
        if denom == 0:
            return 0.0
        corr = np.dot(b1, b2) / (len(b1) * denom)
        return float(max(0.0, min(1.0, corr)))

    def _gradient_cosine_similarity(self, gray1, gray2, mask=None):
        """Compute similarity of gradient direction along strips. Returns [0,1]."""
        if gray1 is None or gray2 is None:
            return 0.0
        # Compute gradients
        gx1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        gx2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
        # Flatten
        v1 = np.stack([gx1.ravel(), gy1.ravel()], axis=1)
        v2 = np.stack([gx2.ravel(), gy2.ravel()], axis=1)
        if mask is not None:
            mask_flat = mask.ravel() > 0
            if not np.any(mask_flat):
                return 0.0
            v1 = v1[mask_flat]
            v2 = v2[mask_flat]
        # Normalize
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = (n1 > 1e-6) & (n2 > 1e-6)
        if not np.any(valid):
            return 0.0
        v1n = v1[valid] / n1[valid][:, None]
        v2n = v2[valid] / n2[valid][:, None]
        cosines = np.sum(v1n * v2n, axis=1)
        cosines = np.clip(cosines, -1.0, 1.0)
        # Map from [-1,1] to [0,1]
        return float(np.mean((cosines + 1.0) / 2.0))
    
    def _compute_curvature(self, points, window=3):
        """Compute discrete curvature."""
        if len(points) < window * 2:
            return np.array([])
        
        curvatures = []
        for i in range(window, len(points) - window):
            p1 = points[i - window]
            p2 = points[i]
            p3 = points[i + window]
            
            # Menger curvature
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a * b * c > 1e-6:
                s = (a + b + c) / 2
                area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                k = 4 * area / (a * b * c)
                curvatures.append(k)
            else:
                curvatures.append(0)
        
        return np.array(curvatures)
    
    def _create_edge_signature(self, points, n_samples=64):
        """Create normalized edge signature."""
        if len(points) < 3:
            return np.zeros((n_samples, 2))
        
        # Resample to fixed number of points
        total_length = cv2.arcLength(points.reshape(-1, 1, 2), False)
        if total_length < 1e-6:
            return np.zeros((n_samples, 2))
        
        # Simple linear interpolation
        indices = np.linspace(0, len(points) - 1, n_samples)
        sampled = np.array([points[int(i)] for i in indices])
        
        # Normalize
        centroid = sampled.mean(axis=0)
        sampled = sampled - centroid
        scale = np.std(sampled)
        if scale > 0:
            sampled = sampled / scale
        
        return sampled
    
    def _edge_color_gradient(self, points, img, mask):
        """Compute color gradient along edge."""
        if len(points) < 5:
            return np.array([0, 0, 0])
        
        colors = []
        for pt in points[::max(1, len(points) // 10)]:  # Sample 10 points
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                if mask[y, x] > 0:
                    colors.append(img[y, x])
        
        if colors:
            colors = np.array(colors, dtype=np.float32)
            return colors.std(axis=0)
        return np.array([0, 0, 0])
    
    def _classify_edge_type(self, points):
        """Classify edge as straight, convex, or concave."""
        if len(points) < 5:
            return 0  # straight
        
        # Fit line
        [vx, vy, x0, y0] = cv2.fitLine(points.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Compute distances to line
        distances = []
        for pt in points:
            # Distance from point to line
            d = abs((pt[1] - y0) * vx - (pt[0] - x0) * vy)
            distances.append(d)
        
        avg_dist = np.mean(distances)
        if avg_dist < 2:
            return 0  # straight
        
        # Check if mostly above or below line
        signed_distances = [(pt[1] - y0) * vx - (pt[0] - x0) * vy for pt in points]
        avg_signed = np.mean(signed_distances)
        
        if avg_signed > 1:
            return 1  # convex
        elif avg_signed < -1:
            return -1  # concave
        return 0
    
    def compute_compatibility(self, tile1_idx: int, dir1: str, tile2_idx: int, dir2: str) -> float:
        """
        Compute compatibility score between two edges.
        Returns score in [0, 1] where 1 = perfect match.
        """
        # Check if edges are opposite directions (required for matching)
        opposite_pairs = {
            ('top', 'bottom'), ('bottom', 'top'),
            ('left', 'right'), ('right', 'left')
        }
        
        if (dir1, dir2) not in opposite_pairs:
            return 0.0
        
        # Pixel-based matching only (contour features disabled)
        tile1 = self.tiles[tile1_idx]
        tile2 = self.tiles[tile2_idx]
        pixel_score = self._pixel_edge_match(tile1, tile2, dir1, dir2)
        if pixel_score is None:
            return 0.0
        return float(pixel_score)
    
    def _pixel_edge_match(self, tile1, tile2, dir1, dir2):
        """
        Compare actual pixel values along edges using multiple metrics.
        This is the REAL test of whether pieces fit together.
        """
        img1 = tile1['img']
        img2 = tile2['img']
        mask1 = tile1['mask']
        mask2 = tile2['mask']
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use config from self (tunable)
        strip_width = self.strip_width
        min_overlap_pixels = self.min_overlap_pixels
        mask_dilate_px = self.mask_dilate_px
        w_border = self.w_border
        w_masked_color = self.w_masked_color
        w_hist = self.w_hist
        w_grad = self.w_grad
        w_ncc = self.w_ncc
        
        # Extract edge strips based on direction
        if dir1 == 'right' and dir2 == 'left':
            # Compare right edge of tile1 with left edge of tile2
            strip1 = img1[:, max(0, w1-strip_width):w1]
            strip2 = img2[:, 0:min(strip_width, w2)]
            mask_strip1 = mask1[:, max(0, w1-strip_width):w1]
            mask_strip2 = mask2[:, 0:min(strip_width, w2)]
            # Extract the actual border pixels (rightmost of tile1, leftmost of tile2)
            border1 = img1[:, -1:] if w1 > 0 else None
            border2 = img2[:, :1] if w2 > 0 else None
            
        elif dir1 == 'left' and dir2 == 'right':
            strip1 = img1[:, 0:min(strip_width, w1)]
            strip2 = img2[:, max(0, w2-strip_width):w2]
            mask_strip1 = mask1[:, 0:min(strip_width, w1)]
            mask_strip2 = mask2[:, max(0, w2-strip_width):w2]
            border1 = img1[:, :1] if w1 > 0 else None
            border2 = img2[:, -1:] if w2 > 0 else None
            
        elif dir1 == 'bottom' and dir2 == 'top':
            strip1 = img1[max(0, h1-strip_width):h1, :]
            strip2 = img2[0:min(strip_width, h2), :]
            mask_strip1 = mask1[max(0, h1-strip_width):h1, :]
            mask_strip2 = mask2[0:min(strip_width, h2), :]
            border1 = img1[-1:, :] if h1 > 0 else None
            border2 = img2[:1, :] if h2 > 0 else None
            
        elif dir1 == 'top' and dir2 == 'bottom':
            strip1 = img1[0:min(strip_width, h1), :]
            strip2 = img2[max(0, h2-strip_width):h2, :]
            mask_strip1 = mask1[0:min(strip_width, h1), :]
            mask_strip2 = mask2[max(0, h2-strip_width):h2, :]
            border1 = img1[:1, :] if h1 > 0 else None
            border2 = img2[-1:, :] if h2 > 0 else None
        else:
            return None
        
        # Resize to match if needed
        if strip1.shape[:2] != strip2.shape[:2]:
            target_h = min(strip1.shape[0], strip2.shape[0])
            target_w = min(strip1.shape[1], strip2.shape[1])
            if target_h < 1 or target_w < 1:
                return None
            strip1 = cv2.resize(strip1, (target_w, target_h))
            strip2 = cv2.resize(strip2, (target_w, target_h))
            mask_strip1 = cv2.resize(mask_strip1, (target_w, target_h))
            mask_strip2 = cv2.resize(mask_strip2, (target_w, target_h))
            if border1 is not None and border2 is not None:
                if border1.shape[0] == border2.shape[0]:
                    pass  # Same height, good
                else:
                    target_len = min(border1.shape[0], border2.shape[0])
                    if dir1 in ['right', 'left']:
                        border1 = cv2.resize(border1, (1, target_len))
                        border2 = cv2.resize(border2, (1, target_len))
                    else:
                        border1 = cv2.resize(border1, (target_len, 1))
                        border2 = cv2.resize(border2, (target_len, 1))
        
        scores = []
        
        # Preprocess mask strips: ensure binary, dilate slightly to allow overlap
        kernel = np.ones((3, 3), np.uint8)
        mask_strip1_d = cv2.dilate((mask_strip1 > 127).astype(np.uint8) * 255, kernel, iterations=mask_dilate_px)
        mask_strip2_d = cv2.dilate((mask_strip2 > 127).astype(np.uint8) * 255, kernel, iterations=mask_dilate_px)
        valid_mask = (mask_strip1_d > 127) & (mask_strip2_d > 127)

        # 1. Border pixel correlation
        if border1 is not None and border2 is not None and border1.size > 0 and border2.size > 0:
            b1_flat = border1.reshape(-1).astype(np.float32)
            b2_flat = border2.reshape(-1).astype(np.float32)
            if len(b1_flat) == len(b2_flat) and len(b1_flat) > 0:
                # Normalized cross-correlation
                b1_norm = b1_flat - b1_flat.mean()
                b2_norm = b2_flat - b2_flat.mean()
                if np.std(b1_norm) > 0 and np.std(b2_norm) > 0:
                    correlation = np.corrcoef(b1_norm, b2_norm)[0, 1]
                    if not np.isnan(correlation):
                        scores.append(max(0.0, min(1.0, correlation)) * w_border)  # Correlation score
        
        # 2. Masked color difference in strips (if overlap small, fallback to histogram)
        if not np.any(valid_mask):
            # Try relaxed overlap with dilated masks
            valid_mask = (mask_strip1_d > 127) & (mask_strip2_d > 127)
        if not np.any(valid_mask):
            # If no overlap, we'll still compute hist similarity on full strips
            pass
        
        diff = np.abs(strip1.astype(np.float32) - strip2.astype(np.float32))
        if diff.ndim == 3:
            diff = diff.mean(axis=2)
        
        masked_diff = diff[valid_mask]
        masked_similarity = 0.0
        if masked_diff.size > 0:
            avg_diff = masked_diff.mean()
            masked_similarity = np.exp(-avg_diff / 40.0)  # 40 is tolerance
            scores.append(masked_similarity * w_masked_color)
        
        # 3. Gradient continuity (edges should flow smoothly)
        gray1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY) if strip1.ndim == 3 else strip1
        gray2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY) if strip2.ndim == 3 else strip2
        
        # Sobel gradients
        if dir1 in ['right', 'left']:
            # Horizontal gradients
            grad1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
            grad2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
        else:
            # Vertical gradients  
            grad1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
            grad2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
        
        grad_diff = np.abs(grad1 - grad2)
        grad_score = np.exp(-grad_diff[valid_mask].mean() / 20.0) if np.any(valid_mask) else 0.2
        scores.append(grad_score * w_grad)

        # 4. Histogram similarity (masked) - robust to lighting
        hist_sim = self._masked_histogram_similarity(strip1, mask_strip1, strip2, mask_strip2, bins=16)
        scores.append(hist_sim * w_hist)

        # 5. NCC on border vectors
        ncc_score = self._ncc_1d_border(border1, border2)
        scores.append(ncc_score * w_ncc)

        # 4. Histogram similarity (masked) - robust to lighting
        hist_sim = self._masked_histogram_similarity(strip1, mask_strip1, strip2, mask_strip2, bins=16)
        scores.append(hist_sim * w_hist)

        # 5. NCC on border vectors
        ncc_score = self._ncc_1d_border(border1, border2)
        scores.append(ncc_score * w_ncc)
        
        # Combine but ensure overall normalization to [0,1]
        if not scores:
            return 0.2
        total = sum(scores)
        # The maximum possible here is sum(weights)=1.0 (or less if border corr not included); normalize by sum of weights present
        weight_sum = 0.0
        for s in [w_border, w_masked_color, w_hist, w_grad, w_ncc]:
            weight_sum += s
        if weight_sum <= 0:
            return float(min(1.0, max(0.0, total)))
        return float(min(1.0, max(0.0, total / weight_sum)))


class PuzzleSolver:
    """Constraint satisfaction solver for puzzle assembly."""
    
    def __init__(self, tiles: List[Dict], rows: int, cols: int):
        self.tiles = tiles
        self.rows = rows
        self.cols = cols
        self.n = len(tiles)
        
        if rows * cols != self.n:
            raise ValueError(f"Grid {rows}x{cols} doesn't match {self.n} tiles")
        
        self.matcher = EdgeMatcher(tiles)
        self.compatibility_cache = {}
        self._precompute_compatibilities()
    
    def _precompute_compatibilities(self):
        """Precompute all edge compatibilities."""
        print("  Computing edge compatibilities...")
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                
                # Horizontal (i on left, j on right)
                h_score = self.matcher.compute_compatibility(i, 'right', j, 'left')
                self.compatibility_cache[(i, j, 'h')] = h_score
                
                # Vertical (i on top, j on bottom)
                v_score = self.matcher.compute_compatibility(i, 'bottom', j, 'top')
                self.compatibility_cache[(i, j, 'v')] = v_score
    
    def get_compatibility(self, tile1: int, tile2: int, direction: str) -> float:
        """Get cached compatibility score."""
        return self.compatibility_cache.get((tile1, tile2, direction), 0.0)
    
    def solve(self, time_limit: float = 60.0) -> Optional[Dict]:
        """
        Solve puzzle using backtracking with edge constraints.
        Returns placement dict or None if no solution found.
        """
        print(f"  Solving {self.rows}x{self.cols} puzzle with {self.n} pieces...")
        
        grid = [[-1] * self.cols for _ in range(self.rows)]
        used = [False] * self.n
        
        # Build candidate lists for each position
        candidates = self._build_candidate_lists()
        
        start_time = time.time()
        best_solution = {'grid': None, 'score': -np.inf}
        
        def backtrack(pos: int, current_score: float) -> bool:
            if time.time() - start_time > time_limit:
                return False
            
            if pos == self.rows * self.cols:
                # Found complete solution
                if current_score > best_solution['score']:
                    best_solution['grid'] = [row[:] for row in grid]
                    best_solution['score'] = current_score
                    print(f"    Found solution with score: {current_score:.3f}")
                return True
            
            r = pos // self.cols
            c = pos % self.cols
            
            # Get candidates for this position
            position_candidates = candidates.get((r, c), list(range(self.n)))
            
            for tile_idx in position_candidates:
                if used[tile_idx]:
                    continue
                
                # Check compatibility with neighbors
                score_delta = 0.0
                
                # Left neighbor
                if c > 0 and grid[r][c-1] != -1:
                    score_delta += self.get_compatibility(grid[r][c-1], tile_idx, 'h')
                
                # Top neighbor
                if r > 0 and grid[r-1][c] != -1:
                    score_delta += self.get_compatibility(grid[r-1][c], tile_idx, 'v')
                
                # Early pruning: if compatibility is too low, skip
                if (c > 0 or r > 0) and score_delta < 0.1:
                    if score_delta < 0.0:
                        continue
                
                # Place tile
                grid[r][c] = tile_idx
                used[tile_idx] = True
                
                # Recurse
                if backtrack(pos + 1, current_score + score_delta):
                    if best_solution['grid'] is not None:
                        # Found a solution, continue to find better ones
                        pass
                
                # Backtrack
                grid[r][c] = -1
                used[tile_idx] = False
            
            return best_solution['grid'] is not None
        
        # Start solving
        backtrack(0, 0.0)
        
        elapsed = time.time() - start_time
        
        if best_solution['grid'] is None:
            print(f"  No solution found in {elapsed:.1f}s")
            # Return greedy solution
            return self._greedy_solve()
        else:
            print(f"  Solution found in {elapsed:.1f}s with score {best_solution['score']:.3f}")
            
            # Convert grid to placement dict
            placement = {}
            for r in range(self.rows):
                for c in range(self.cols):
                    placement[f"{r}_{c}"] = best_solution['grid'][r][c]
            
            return {
                'placement_map': placement,
                'grid': best_solution['grid'],
                'score': best_solution['score'],
                'method': 'backtracking'
            }
    
    def _build_candidate_lists(self) -> Dict:
        """Build ordered candidate lists for each position."""
        candidates = {}
        
        # For each position, rank tiles by average compatibility with all others
        for r in range(self.rows):
            for c in range(self.cols):
                scores = []
                for tile_idx in range(self.n):
                    # Average compatibility with all other tiles
                    avg_score = 0.0
                    count = 0
                    for other_idx in range(self.n):
                        if tile_idx == other_idx:
                            continue
                        avg_score += self.get_compatibility(tile_idx, other_idx, 'h')
                        avg_score += self.get_compatibility(tile_idx, other_idx, 'v')
                        count += 2
                    if count > 0:
                        avg_score /= count
                    scores.append((avg_score, tile_idx))
                
                # Sort by score descending
                scores.sort(reverse=True)
                candidates[(r, c)] = [tile_idx for _, tile_idx in scores]
        
        return candidates
    
    def _greedy_solve(self) -> Dict:
        """Fallback greedy solver."""
        print("  Falling back to greedy solver...")
        
        grid = [[-1] * self.cols for _ in range(self.rows)]
        used = [False] * self.n
        
        for r in range(self.rows):
            for c in range(self.cols):
                best_tile = -1
                best_score = -np.inf
                
                for tile_idx in range(self.n):
                    if used[tile_idx]:
                        continue
                    
                    score = 0.0
                    if c > 0 and grid[r][c-1] != -1:
                        score += self.get_compatibility(grid[r][c-1], tile_idx, 'h')
                    if r > 0 and grid[r-1][c] != -1:
                        score += self.get_compatibility(grid[r-1][c], tile_idx, 'v')
                    
                    if score > best_score:
                        best_score = score
                        best_tile = tile_idx
                
                if best_tile != -1:
                    grid[r][c] = best_tile
                    used[best_tile] = True
        
        placement = {}
        for r in range(self.rows):
            for c in range(self.cols):
                placement[f"{r}_{c}"] = grid[r][c]
        
        return {
            'placement_map': placement,
            'grid': grid,
            'score': 0.0,
            'method': 'greedy'
        }
