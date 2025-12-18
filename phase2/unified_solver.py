from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import time
from itertools import permutations
from collections import deque


def _prepare_tile_image(tile: Dict) -> np.ndarray:
    # Extract image from tile dictionary
    return tile["img"]


def _extract_borders(piece: np.ndarray, strip_width: int = 1) -> Dict[int, np.ndarray]:
    # Extract borders from each side (top, right, bottom, left) with gradient info
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    sw = min(strip_width, h // 2, w // 2)

    def make_grad_patch_enhanced(patch_lab: np.ndarray) -> np.ndarray:
        # Create feature patch with color, gradient, and laplacian
        patch_bgr = cv2.cvtColor(patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        patch_gray = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_gray = cv2.GaussianBlur(patch_gray, (3, 3), 0)
        gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)[..., None]
        grad_dir = cv2.phase(gx, gy, angleInDegrees=True)[..., None]
        lap = cv2.Laplacian(patch_gray, cv2.CV_32F)[..., None]
        return np.concatenate([patch_lab, grad_mag, grad_dir, lap], axis=2)

    return {
        0: make_grad_patch_enhanced(lab[0:sw, :, :]),
        1: make_grad_patch_enhanced(lab[:, w - sw : w, :]),
        2: make_grad_patch_enhanced(lab[h - sw : h, :, :]),
        3: make_grad_patch_enhanced(lab[:, 0:sw, :]),
    }


def _normalize_strip_2d(strip: np.ndarray) -> np.ndarray:
    # Normalize each channel to zero mean and unit variance
    arr = strip.astype(np.float32)
    for ch in range(arr.shape[2]):
        m, sd = arr[..., ch].mean(), arr[..., ch].std()
        arr[..., ch] = (arr[..., ch] - m) / (sd if sd > 1e-6 else 1.0)
    return arr


def _border_distance_2d(
    stripA: np.ndarray,
    stripB: np.ndarray,
    sideA: int,
    sideB: int,
    p: float = 0.3,
    q: float = 1 / 16,
    w_color: float = 0.4,
    w_grad_mag: float = 0.2,
    w_grad_dir: float = 0.2,
    w_lap: float = 0.4,
) -> float:
    # Calculate distance between two borders using color and gradient features
    def orient(strip: np.ndarray, side: int) -> np.ndarray:
        return _normalize_strip_2d(np.transpose(strip, (1, 0, 2)) if side in (1, 3) else strip)

    a, b = orient(stripA, sideA), orient(stripB, sideB)
    if a.size == 0 or b.size == 0:
        return 1e9
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    def dist(x: np.ndarray, y: np.ndarray) -> float:
        d_color = np.sum(np.abs(x[..., :3] - y[..., :3]) ** p)
        d_grad_mag = np.sum(np.abs(x[..., 3:4] - y[..., 3:4]) ** p)
        d_grad_dir = np.sum(np.abs(x[..., 4:5] - y[..., 4:5]) ** p)
        d_lap = np.sum(np.abs(x[..., 5:8] - y[..., 5:8]) ** p)
        total = w_color * d_color + w_grad_mag * d_grad_mag + w_grad_dir * d_grad_dir + w_lap * d_lap
        return total ** (q / p)

    return float(dist(a, b))


def _build_compatibility(pieces: List[np.ndarray], strip_width: int = 1) -> Dict[int, np.ndarray]:
    # Build compatibility matrix for all piece pairs across all sides
    n = len(pieces)
    borders = [_extract_borders(p, strip_width) for p in pieces]
    compat = {s: np.full((n, n), 1e9, dtype=np.float32) for s in range(4)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            compat[0][i, j] = _border_distance_2d(borders[i][0], borders[j][2], 0, 2)
            compat[1][i, j] = _border_distance_2d(borders[i][1], borders[j][3], 1, 3)
            compat[2][i, j] = _border_distance_2d(borders[i][2], borders[j][0], 2, 0)
            compat[3][i, j] = _border_distance_2d(borders[i][3], borders[j][1], 3, 1)
    return compat


def _opposite(side: int) -> int:
    # Get opposite side (0->2, 1->3, 2->0, 3->1)
    return (side + 2) % 4


def _best_partner_for(i: int, side: int, compat: Dict[int, np.ndarray]) -> int:
    # Find best matching piece for given piece and side
    return int(np.argmin(compat[side][i]))


def _is_best_buddy(i: int, side: int, j: int, compat: Dict[int, np.ndarray]) -> bool:
    # Check if two pieces are best buddies (mutually best partners)
    if i == j:
        return False
    bj = _best_partner_for(i, side, compat)
    if bj != j:
        return False
    opp = _opposite(side)
    bi = _best_partner_for(j, opp, compat)
    return bi == i


def _placer(n: int, grid_n: int, compat: Dict[int, np.ndarray], seed_placement=None, seed_center: bool = True):
    # Place pieces on grid starting from seed and filling remaining slots
    placement = [-1] * n
    used = [False] * n

    if seed_placement:
        # Position seed pieces in center or top-left
        seed_pos = list(seed_placement.keys())
        rs = [p // grid_n for p in seed_pos]
        cs = [p % grid_n for p in seed_pos]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        seed_h = rmax - rmin + 1
        seed_w = cmax - cmin + 1
        top = (grid_n - seed_h) // 2 if seed_center else 0
        left = (grid_n - seed_w) // 2 if seed_center else 0
        for pos_old, pid in seed_placement.items():
            r_old, c_old = pos_old // grid_n, pos_old % grid_n
            r_new = top + (r_old - rmin)
            c_new = left + (c_old - cmin)
            if 0 <= r_new < grid_n and 0 <= c_new < grid_n:
                pos_new = r_new * grid_n + c_new
                placement[pos_new] = pid
                used[pid] = True
    else:
        # Start with random seed piece
        seed_pid = np.random.randint(0, n)
        seed_pos = np.random.choice(range(n))
        placement[seed_pos] = seed_pid
        used[seed_pid] = True

    def get_neighbors(pos: int):
        # Get filled neighbors of empty position
        r, c = pos // grid_n, pos % grid_n
        neighbors = []
        if r > 0 and placement[pos - grid_n] != -1:
            neighbors.append((pos - grid_n, 2))
        if r < grid_n - 1 and placement[pos + grid_n] != -1:
            neighbors.append((pos + grid_n, 0))
        if c > 0 and placement[pos - 1] != -1:
            neighbors.append((pos - 1, 1))
        if c < grid_n - 1 and placement[pos + 1] != -1:
            neighbors.append((pos + 1, 3))
        return neighbors

    slots_filled = sum(1 for x in placement if x != -1)
    while slots_filled < n:
        # Fill empty slots starting with most constrained positions
        empty_slots = []
        for pos in range(n):
            if placement[pos] != -1:
                continue
            neighs = get_neighbors(pos)
            if neighs:
                empty_slots.append((-len(neighs), pos, neighs))
        if not empty_slots:
            pos = placement.index(-1)
            empty_slots = [(0, pos, [])]

        empty_slots.sort()
        chosen = None

        for _, slot_pos, neighs in empty_slots:
            # Prioritize pieces with best buddy matches
            candidates = []
            for pid in range(n):
                if used[pid]:
                    continue
                bb_count = 0
                compat_sum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    if _is_best_buddy(neigh_pid, neigh_side, pid, compat):
                        bb_count += 1
                    compat_sum += compat[neigh_side][neigh_pid, pid]
                if bb_count > 0:
                    candidates.append((bb_count, compat_sum, slot_pos, pid))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
                break

        if chosen is None:
            # Fallback: pick piece with lowest average compatibility
            _, slot_pos, neighs = empty_slots[0]
            best_val = 1e18
            best_pid = None
            for pid in range(n):
                if used[pid]:
                    continue
                ssum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    ssum += compat[neigh_side][neigh_pid, pid]
                avg = ssum / max(1, len(neighs))
                if avg < best_val:
                    best_val = avg
                    best_pid = pid
            chosen = (0, best_val, slot_pos, best_pid)

        _, _, slot_pos, chosen_pid = chosen
        placement[slot_pos] = chosen_pid
        used[chosen_pid] = True
        slots_filled += 1

    return placement


def _segmenter(placement: List[int], grid_n: int, compat: Dict[int, np.ndarray]):
    # Find connected components of pieces linked by best buddy relationships
    n_slots = len(placement)
    visited = [False] * n_slots
    segments = []

    def neighbors(pos: int):
        # Get neighboring positions (up, down, left, right)
        r = pos // grid_n
        c = pos % grid_n
        if c > 0:
            yield pos - 1, 3
        if c < grid_n - 1:
            yield pos + 1, 1
        if r > 0:
            yield pos - grid_n, 0
        if r < grid_n - 1:
            yield pos + grid_n, 2

    for pos in range(n_slots):
        # BFS to find connected components
        if visited[pos]:
            continue
        queue = deque([pos])
        comp = []
        visited[pos] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            pu = placement[u]
            for v, side_of_u in neighbors(u):
                if visited[v]:
                    continue
                pv = placement[v]
                if _is_best_buddy(pu, side_of_u, pv, compat):
                    visited[v] = True
                    queue.append(v)
        if comp:
            segments.append(comp)
    return segments


def _compute_best_buddies_score(placement: List[int], grid_n: int, compat: Dict[int, np.ndarray]) -> float:
    # Calculate fraction of edges that are best buddy pairs
    n_slots = len(placement)
    bb_count = 0
    total_adj = 0
    for pos in range(n_slots):
        r = pos // grid_n
        c = pos % grid_n
        pid = placement[pos]
        if c < grid_n - 1:
            np0 = pos + 1
            pid2 = placement[np0]
            total_adj += 1
            if _is_best_buddy(pid, 1, pid2, compat):
                bb_count += 1
        if r < grid_n - 1:
            np1 = pos + grid_n
            pid2 = placement[np1]
            total_adj += 1
            if _is_best_buddy(pid, 2, pid2, compat):
                bb_count += 1
    if total_adj == 0:
        return 0.0
    return bb_count / total_adj


def _shifter(initial_placement: List[int], grid_n: int, compat: Dict[int, np.ndarray], max_iters: int = 8, swap_pass: bool = True):
    # Improve placement by re-solving segments and swapping pieces
    n_slots = len(initial_placement)
    current = initial_placement.copy()
    best_score = _compute_best_buddies_score(current, grid_n, compat)

    for _ in range(max_iters):
        # Find connected best buddy segments
        segments = _segmenter(current, grid_n, compat)
        if not segments:
            break
        segments.sort(key=lambda x: -len(x))
        improved = False

        for seg in segments:
            if not seg:
                continue
            # Re-place segment pieces
            seed_map = {pos: current[pos] for pos in seg}
            placement_new = _placer(n_slots, grid_n, compat, seed_placement=seed_map)
            score_new = _compute_best_buddies_score(placement_new, grid_n, compat)
            if score_new > best_score + 1e-9:
                current = placement_new
                best_score = score_new
                improved = True
                break

        if not improved and swap_pass:
            # Try swapping piece positions
            for pos1 in range(n_slots):
                for pos2 in range(pos1 + 1, n_slots):
                    new_p = current.copy()
                    new_p[pos1], new_p[pos2] = new_p[pos2], new_p[pos1]
                    score_swap = _compute_best_buddies_score(new_p, grid_n, compat)
                    if score_swap > best_score + 1e-9:
                        current = new_p
                        best_score = score_swap
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return current, best_score


def _solve_bruteforce(pieces: List[np.ndarray], compat: Dict[int, np.ndarray], grid_n: int):
    # Exhaustive search for small grids (2x2)
    n = grid_n * grid_n
    best_perm = None
    best_score = 1e12
    for perm in permutations(range(n)):
        score = 0.0
        valid = True
        for pos, pid in enumerate(perm):
            r = pos // grid_n
            c = pos % grid_n
            if c > 0:
                left_pid = perm[pos - 1]
                score += compat[1][left_pid, pid]
                if score >= best_score:
                    valid = False
                    break
            if r > 0:
                top_pid = perm[pos - grid_n]
                score += compat[2][top_pid, pid]
                if score >= best_score:
                    valid = False
                    break
        if not valid:
            continue
        if score < best_score:
            best_score = score
            best_perm = perm
    return list(best_perm), best_score


class PuzzleSolver:
    # Solve puzzle using best-buddies algorithm with multiple seeded attempts

    def __init__(self, tiles: List[Dict], rows: int, cols: int, strip_width: int = 1, seeds: Optional[int] = None, shifter_iters: int = 8, beam_width: Optional[int] = None):
        self.tiles = tiles
        self.rows = rows
        self.cols = cols
        self.n = len(tiles)
        if rows * cols != self.n:
            raise ValueError(f"Grid {rows}x{cols} doesn't match {self.n} tiles")
        self.grid_n = rows
        self.strip_width = strip_width
        # Adaptive seed count based on grid size
        if seeds is None:
            self.seeds = {2: 5, 4: 10, 8: 20}.get(self.grid_n, 10)
        else:
            self.seeds = seeds
        self.shifter_iters = shifter_iters
        # Adaptive beam width based on grid size
        if beam_width is None:
            self.beam_width = {2: 1, 4: 3, 8: 5}.get(self.grid_n, 3)
        else:
            self.beam_width = max(1, beam_width)
        self.pieces = [_prepare_tile_image(t) for t in tiles]

    def solve(self, time_limit: float = 60.0, beam_width: int = 0) -> Optional[Dict]:
        # Solve puzzle with multiple attempts and return best placement
        start = time.time()
        compat = _build_compatibility(self.pieces, strip_width=self.strip_width)
        effective_beam = beam_width if beam_width > 0 else self.beam_width

        if self.grid_n == 2:
            # Use brute force for 2x2 puzzles
            order, _ = _solve_bruteforce(self.pieces, compat, self.grid_n)
            placement = order
            best_bb = _compute_best_buddies_score(order, self.grid_n, compat)
        else:
            # Try multiple seeds with shifting refinement
            candidates = []
            best_placement = None
            best_bb = -1.0
            seed_id = 0
            while seed_id < self.seeds and (time.time() - start) < time_limit:
                init_placement = _placer(self.n, self.grid_n, compat, seed_placement=None)
                bb0 = _compute_best_buddies_score(init_placement, self.grid_n, compat)
                placement_after_shifter, bb_sh = _shifter(init_placement, self.grid_n, compat, max_iters=self.shifter_iters)
                final_placement = placement_after_shifter if bb_sh >= bb0 else init_placement
                final_bb = bb_sh if bb_sh >= bb0 else bb0
                candidates.append((final_bb, final_placement))
                if final_bb > best_bb:
                    best_bb = final_bb
                    best_placement = final_placement
                seed_id += 1

            # Try extra seeds for difficult puzzles
            if best_bb < 0.60 or (time.time() - start) < time_limit:
                extra_seeds = 5
                for _ in range(extra_seeds):
                    if (time.time() - start) >= time_limit:
                        break
                    init_placement = _placer(self.n, self.grid_n, compat, seed_placement=None)
                    bb0 = _compute_best_buddies_score(init_placement, self.grid_n, compat)
                    placement_after_shifter, bb_sh = _shifter(init_placement, self.grid_n, compat, max_iters=self.shifter_iters)
                    final_placement = placement_after_shifter if bb_sh >= bb0 else init_placement
                    final_bb = bb_sh if bb_sh >= bb0 else bb0
                    candidates.append((final_bb, final_placement))
                    if final_bb > best_bb:
                        best_bb = final_bb
                        best_placement = final_placement

            # Keep top candidates and select best
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:effective_beam]
                best_bb, best_placement = candidates[0]

            placement = best_placement if best_placement is not None else init_placement

        # Convert placement to string keys
        placement_map = {}
        for pos, pid in enumerate(placement):
            r = pos // self.grid_n
            c = pos % self.grid_n
            placement_map[f"{r}_{c}"] = pid

        elapsed = time.time() - start
        return {
            "placement_map": placement_map,
            "order": placement,
            "score": float(best_bb),
            "method": "best_buddies",
            "time": elapsed,
        }
