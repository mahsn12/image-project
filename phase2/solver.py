# phase2/solver.py
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import time
from phase2.features import distance_to_similarity_matrix

def build_combined_similarity(horizontal_distance: np.ndarray, vertical_distance: np.ndarray, global_similarity: np.ndarray,
                              object_similarity_h: np.ndarray = None, object_similarity_v: np.ndarray = None,
                              edge_similarity_h: np.ndarray = None, edge_similarity_v: np.ndarray = None,
                              alpha: float = 0.8, gamma: float = 0.0, delta: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    beta = 1.0 - alpha - gamma - delta
    if beta < -1e-12:
        raise ValueError("alpha + gamma + delta must be <= 1.0")
    H_sim_border = distance_to_similarity_matrix(horizontal_distance)
    V_sim_border = distance_to_similarity_matrix(vertical_distance)
    H_comb = alpha * H_sim_border + max(0.0,beta) * global_similarity
    V_comb = alpha * V_sim_border + max(0.0,beta) * global_similarity
    if gamma > 0.0 and object_similarity_h is not None and object_similarity_v is not None:
        H_comb = H_comb * (1.0 - gamma) + gamma * object_similarity_h
        V_comb = V_comb * (1.0 - gamma) + gamma * object_similarity_v
    if delta > 0.0 and edge_similarity_h is not None and edge_similarity_v is not None:
        H_comb = H_comb * (1.0 - delta) + delta * edge_similarity_h
        V_comb = V_comb * (1.0 - delta) + delta * edge_similarity_v
    return np.clip(H_comb,0.0,1.0), np.clip(V_comb,0.0,1.0)

def save_heatmap(matrix: np.ndarray, output_path: str, title_text: str):
    plt.figure(figsize=(6,5))
    plt.imshow(matrix, aspect='auto')
    plt.colorbar()
    plt.title(title_text)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def compute_edge_matches(horizontal_similarity: np.ndarray, vertical_similarity: np.ndarray, top_k: int = 4) -> Dict[str, List[List[int]]]:
    tile_count = horizontal_similarity.shape[0]
    right_candidates = []; left_candidates = []; bottom_candidates = []; top_candidates = []
    for tile_idx in range(tile_count):
        right_candidates.append(list(np.argsort(-horizontal_similarity[tile_idx,:])[:top_k].astype(int)))
        left_candidates.append(list(np.argsort(-horizontal_similarity[:,tile_idx])[:top_k].astype(int)))
        bottom_candidates.append(list(np.argsort(-vertical_similarity[tile_idx,:])[:top_k].astype(int)))
        top_candidates.append(list(np.argsort(-vertical_similarity[:,tile_idx])[:top_k].astype(int)))
    return {"right_of": right_candidates, "left_of": left_candidates, "bottom_of": bottom_candidates, "top_of": top_candidates}

def place_tiles_exact(horizontal_similarity: np.ndarray, vertical_similarity: np.ndarray, rows: int, cols: int, top_k: int = 4, time_limit_sec: float = 8.0):
    tile_count = horizontal_similarity.shape[0]
    if rows*cols != tile_count:
        raise ValueError("rows*cols != n_tiles")
    start_time = time.time()
    positions = [(r,c) for r in range(rows) for c in range(cols)]
    grid = [[None]*cols for _ in range(rows)]
    used = [False]*tile_count
    best = {"score": -1.0, "grid": None}
    top_right_candidates = [list(np.argsort(-horizontal_similarity[i,:])[:top_k]) for i in range(tile_count)]
    top_bottom_candidates = [list(np.argsort(-vertical_similarity[i,:])[:top_k]) for i in range(tile_count)]
    def ub(level, current_score):
        remaining = tile_count - level
        return current_score + remaining * max(np.max(horizontal_similarity), np.max(vertical_similarity))
    def dfs(level, current_score):
        if time.time() - start_time > time_limit_sec:
            return
        if level == len(positions):
            if current_score > best["score"]:
                best["score"] = current_score
                best["grid"] = [row[:] for row in grid]
            return
        r,c = positions[level]
        if level == 0:
            candidates = [i for i in range(tile_count) if not used[i]]
        else:
            if c>0:
                left_idx = grid[r][c-1]
                candidates = [x for x in top_right_candidates[left_idx] if not used[x]]
                if not candidates: candidates = [i for i in range(tile_count) if not used[i]]
            elif r>0:
                top_idx = grid[r-1][c]
                candidates = [x for x in top_bottom_candidates[top_idx] if not used[x]]
                if not candidates: candidates = [i for i in range(tile_count) if not used[i]]
            else:
                candidates = [i for i in range(tile_count) if not used[i]]
        scored = []
        for cand in candidates:
            if used[cand]: continue
            s = 0.0
            if c>0: s += horizontal_similarity[grid[r][c-1], cand]
            if r>0: s += vertical_similarity[grid[r-1][c], cand]
            scored.append((s, cand))
        scored.sort(key=lambda x:x[0], reverse=True)
        max_search = max(6, top_k*2)
        for score_val, cand in scored[:max_search]:
            grid[r][c] = int(cand); used[cand] = True
            new_score = current_score + score_val
            if ub(level+1, new_score) > best["score"]:
                dfs(level+1, new_score)
            used[cand] = False; grid[r][c] = None
    dfs(0,0.0)
    if best["grid"] is None:
        raise RuntimeError("Exact solver timed out or failed")
    placement_map = {f"{r}_{c}": int(best["grid"][r][c]) for r in range(rows) for c in range(cols)}
    return {"grid": best["grid"], "placement_map": placement_map, "weak_edges": [], "backtracking_used": True}

def _backtracking_try(horizontal_similarity: np.ndarray, vertical_similarity: np.ndarray, rows: int, cols: int, time_limit_sec: float = 8.0):
    tile_count = horizontal_similarity.shape[0]
    positions = [(r,c) for r in range(rows) for c in range(cols)]
    start_time = time.time()
    used = [False]*tile_count; grid = [[None]*cols for _ in range(rows)]
    best_sums = [float(np.max(horizontal_similarity[i,:]) + np.max(vertical_similarity[i,:]) + np.max(horizontal_similarity[:,i]) + np.max(vertical_similarity[:,i])) for i in range(tile_count)]
    start_idx = int(np.argmin(best_sums))
    def score_local(r,c,cand):
        s=0.0; cnt=0
        if c>0: s+=horizontal_similarity[grid[r][c-1], cand]; cnt+=1
        if r>0: s+=vertical_similarity[grid[r-1][c], cand]; cnt+=1
        return (s/cnt) if cnt>0 else 0.0
    def dfs(pos_idx):
        if time.time() - start_time > time_limit_sec:
            return None
        if pos_idx >= len(positions):
            placement_map = {f"{r}_{c}": int(grid[r][c]) for r,c in positions}
            return {"grid": [row[:] for row in grid], "placement_map": placement_map, "weak_edges": [], "backtracking_used": True}
        r,c = positions[pos_idx]
        cand_list = [start_idx] if pos_idx==0 else [i for i in range(tile_count) if not used[i]]
        scored = []
        for cand in cand_list:
            scored.append((score_local(r,c,cand), cand))
        scored.sort(key=lambda x:x[0], reverse=True)
        for _, cand in scored:
            if c>0 and horizontal_similarity[grid[r][c-1], cand] < 0.03: continue
            if r>0 and vertical_similarity[grid[r-1][c], cand] < 0.03: continue
            grid[r][c] = int(cand); used[cand] = True
            res = dfs(pos_idx+1)
            if res is not None: return res
            used[cand] = False; grid[r][c] = None
        return None
    return dfs(0)

def place_tiles_with_edge_checks(horizontal_similarity: np.ndarray, vertical_similarity: np.ndarray, rows: int, cols: int,
                                 top_k: int = 4, min_edge_score: float = 0.12, use_backtracking: bool = True,
                                 time_limit_sec: float = 6.0) -> dict:
    tile_count = horizontal_similarity.shape[0]
    if rows*cols != tile_count: raise ValueError("rows*cols != number of tiles")
    try:
        if tile_count <= 16:
            return place_tiles_exact(horizontal_similarity, vertical_similarity, rows, cols, top_k=top_k, time_limit_sec=time_limit_sec)
    except Exception:
        pass
    matches = compute_edge_matches(horizontal_similarity, vertical_similarity, top_k=top_k)
    right_of = matches["right_of"]; left_of = matches["left_of"]
    bottom_of = matches["bottom_of"]; top_of = matches["top_of"]
    used = set(); grid = [[None]*cols for _ in range(rows)]
    best_sums = [float(np.max(horizontal_similarity[i,:]) + np.max(vertical_similarity[i,:]) + np.max(horizontal_similarity[:,i]) + np.max(vertical_similarity[:,i])) for i in range(tile_count)]
    start_tile = int(np.argmin(best_sums))
    grid[0][0] = start_tile; used.add(start_tile)
    for r in range(rows):
        for c in range(cols):
            if r==0 and c==0: continue
            candidates = [i for i in range(tile_count) if i not in used]
            viable = []
            for cand in candidates:
                ok = True; score_acc=0.0; cnt=0
                if c>0:
                    left_idx = grid[r][c-1]
                    cond1 = (cand in right_of[left_idx]) or (float(horizontal_similarity[left_idx, cand]) >= min_edge_score)
                    cond2 = (left_idx in left_of[cand]) or (float(horizontal_similarity[left_idx, cand]) >= min_edge_score)
                    if not (cond1 and cond2): ok=False
                    score_acc += float(horizontal_similarity[left_idx, cand]); cnt+=1
                if r>0:
                    top_idx = grid[r-1][c]
                    cond1v = (cand in bottom_of[top_idx]) or (float(vertical_similarity[top_idx, cand]) >= min_edge_score)
                    cond2v = (top_idx in top_of[cand]) or (float(vertical_similarity[top_idx, cand]) >= min_edge_score)
                    if not (cond1v and cond2v): ok=False
                    score_acc += float(vertical_similarity[top_idx, cand]); cnt+=1
                avg = (score_acc/cnt) if cnt>0 else float((np.mean(horizontal_similarity[cand,:]) + np.mean(vertical_similarity[cand,:]))/2.0)
                viable.append((ok, avg, cand))
            strict = [t for t in viable if t[0]]
            if strict:
                strict.sort(key=lambda x:x[1], reverse=True); chosen = strict[0][2]
            else:
                viable.sort(key=lambda x:x[1], reverse=True); chosen = viable[0][2]
            grid[r][c] = int(chosen); used.add(chosen)
    placement_map = {f"{r}_{c}": int(grid[r][c]) for r in range(rows) for c in range(cols)}
    placement = {"grid": grid, "placement_map": placement_map}
    weak = []
    for r in range(rows):
        for c in range(cols):
            cur = grid[r][c]
            if cur is None: continue
            if c+1 < cols:
                right = grid[r][c+1]
                if float(horizontal_similarity[cur, right]) < min_edge_score:
                    weak.append({"pos": f"{r}_{c}", "dir": "right", "score": float(horizontal_similarity[cur, right]), "pair":[int(cur), int(right)]})
            if r+1 < rows:
                bottom = grid[r+1][c]
                if float(vertical_similarity[cur, bottom]) < min_edge_score:
                    weak.append({"pos": f"{r}_{c}", "dir": "bottom", "score": float(vertical_similarity[cur, bottom]), "pair":[int(cur), int(bottom)]})
    placement["weak_edges"] = weak; placement["backtracking_used"] = False
    if use_backtracking and len(weak) > 0:
        try:
            bt = _backtracking_try(horizontal_similarity, vertical_similarity, rows, cols, time_limit_sec=time_limit_sec)
            if bt is not None:
                bt["backtracking_used"] = True
                return bt
        except Exception:
            pass
    return placement
