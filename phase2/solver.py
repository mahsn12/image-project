# phase2/solver.py
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import time
from phase2.features import distance_to_similarity_matrix

def build_combined_similarity(H_dist: np.ndarray, V_dist: np.ndarray, global_sim: np.ndarray,
                              object_sim_h: np.ndarray = None, object_sim_v: np.ndarray = None,
                              edge_sim_h: np.ndarray = None, edge_sim_v: np.ndarray = None,
                              alpha: float = 0.8, gamma: float = 0.0, delta: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    beta = 1.0 - alpha - gamma - delta
    if beta < -1e-12:
        raise ValueError("alpha + gamma + delta must be <= 1.0")
    H_sim_border = distance_to_similarity_matrix(H_dist)
    V_sim_border = distance_to_similarity_matrix(V_dist)
    H_comb = alpha * H_sim_border + max(0.0,beta) * global_sim
    V_comb = alpha * V_sim_border + max(0.0,beta) * global_sim
    if gamma > 0.0 and object_sim_h is not None and object_sim_v is not None:
        H_comb = H_comb * (1.0 - gamma) + gamma * object_sim_h
        V_comb = V_comb * (1.0 - gamma) + gamma * object_sim_v
    if delta > 0.0 and edge_sim_h is not None and edge_sim_v is not None:
        H_comb = H_comb * (1.0 - delta) + delta * edge_sim_h
        V_comb = V_comb * (1.0 - delta) + delta * edge_sim_v
    return np.clip(H_comb,0.0,1.0), np.clip(V_comb,0.0,1.0)

def save_heatmap(mat: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(6,5))
    plt.imshow(mat, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def compute_edge_matches(H_sim: np.ndarray, V_sim: np.ndarray, top_k: int = 4) -> Dict[str, List[List[int]]]:
    n = H_sim.shape[0]
    right_of=[]; left_of=[]; bottom_of=[]; top_of=[]
    for i in range(n):
        right_of.append(list(np.argsort(-H_sim[i,:])[:top_k].astype(int)))
        left_of.append(list(np.argsort(-H_sim[:,i])[:top_k].astype(int)))
        bottom_of.append(list(np.argsort(-V_sim[i,:])[:top_k].astype(int)))
        top_of.append(list(np.argsort(-V_sim[:,i])[:top_k].astype(int)))
    return {"right_of": right_of, "left_of": left_of, "bottom_of": bottom_of, "top_of": top_of}

def place_tiles_exact(H_sim: np.ndarray, V_sim: np.ndarray, rows: int, cols: int, top_k: int = 4, time_limit_sec: float = 8.0):
    n = H_sim.shape[0]
    if rows*cols != n:
        raise ValueError("rows*cols != n_tiles")
    start = time.time()
    positions = [(r,c) for r in range(rows) for c in range(cols)]
    grid = [[None]*cols for _ in range(rows)]
    used = [False]*n
    best = {"score": -1.0, "grid": None}
    top_right = [list(np.argsort(-H_sim[i,:])[:top_k]) for i in range(n)]
    top_bottom = [list(np.argsort(-V_sim[i,:])[:top_k]) for i in range(n)]
    def ub(level, current_score):
        rem = n - level
        return current_score + rem * max(np.max(H_sim), np.max(V_sim))
    def dfs(level, current_score):
        if time.time() - start > time_limit_sec:
            return
        if level == len(positions):
            if current_score > best["score"]:
                best["score"] = current_score
                best["grid"] = [row[:] for row in grid]
            return
        r,c = positions[level]
        if level == 0:
            candidates = [i for i in range(n) if not used[i]]
        else:
            if c>0:
                left_idx = grid[r][c-1]
                candidates = [x for x in top_right[left_idx] if not used[x]]
                if not candidates: candidates = [i for i in range(n) if not used[i]]
            elif r>0:
                top_idx = grid[r-1][c]
                candidates = [x for x in top_bottom[top_idx] if not used[x]]
                if not candidates: candidates = [i for i in range(n) if not used[i]]
            else:
                candidates = [i for i in range(n) if not used[i]]
        scored = []
        for cand in candidates:
            if used[cand]: continue
            s = 0.0
            if c>0: s += H_sim[grid[r][c-1], cand]
            if r>0: s += V_sim[grid[r-1][c], cand]
            scored.append((s, cand))
        scored.sort(key=lambda x:x[0], reverse=True)
        M = max(6, top_k*2)
        for sc, cand in scored[:M]:
            grid[r][c] = int(cand); used[cand] = True
            new_score = current_score + sc
            if ub(level+1, new_score) > best["score"]:
                dfs(level+1, new_score)
            used[cand] = False; grid[r][c] = None
    dfs(0,0.0)
    if best["grid"] is None:
        raise RuntimeError("Exact solver timed out or failed")
    placement_map = {f"{r}_{c}": int(best["grid"][r][c]) for r in range(rows) for c in range(cols)}
    return {"grid": best["grid"], "placement_map": placement_map, "weak_edges": [], "backtracking_used": True}

def _backtracking_try(H_sim: np.ndarray, V_sim: np.ndarray, rows: int, cols: int, time_limit_sec: float = 8.0):
    n = H_sim.shape[0]
    positions = [(r,c) for r in range(rows) for c in range(cols)]
    start = time.time()
    used = [False]*n; grid = [[None]*cols for _ in range(rows)]
    best_sums = [float(np.max(H_sim[i,:]) + np.max(V_sim[i,:]) + np.max(H_sim[:,i]) + np.max(V_sim[:,i])) for i in range(n)]
    start_idx = int(np.argmin(best_sums))
    def score_local(r,c,cand):
        s=0.0; cnt=0
        if c>0: s+=H_sim[grid[r][c-1], cand]; cnt+=1
        if r>0: s+=V_sim[grid[r-1][c], cand]; cnt+=1
        return (s/cnt) if cnt>0 else 0.0
    def dfs(pos_idx):
        if time.time() - start > time_limit_sec:
            return None
        if pos_idx >= len(positions):
            placement_map = {f"{r}_{c}": int(grid[r][c]) for r,c in positions}
            return {"grid": [row[:] for row in grid], "placement_map": placement_map, "weak_edges": [], "backtracking_used": True}
        r,c = positions[pos_idx]
        cand_list = [start_idx] if pos_idx==0 else [i for i in range(n) if not used[i]]
        scored = []
        for cand in cand_list:
            scored.append((score_local(r,c,cand), cand))
        scored.sort(key=lambda x:x[0], reverse=True)
        for _, cand in scored:
            if c>0 and H_sim[grid[r][c-1], cand] < 0.03: continue
            if r>0 and V_sim[grid[r-1][c], cand] < 0.03: continue
            grid[r][c] = int(cand); used[cand] = True
            res = dfs(pos_idx+1)
            if res is not None: return res
            used[cand] = False; grid[r][c] = None
        return None
    return dfs(0)

def place_tiles_with_edge_checks(H_sim: np.ndarray, V_sim: np.ndarray, rows: int, cols: int,
                                 top_k: int = 4, min_edge_score: float = 0.12, use_backtracking: bool = True,
                                 time_limit_sec: float = 6.0) -> dict:
    n = H_sim.shape[0]
    if rows*cols != n: raise ValueError("rows*cols != number of tiles")
    try:
        if n <= 16:
            return place_tiles_exact(H_sim, V_sim, rows, cols, top_k=top_k, time_limit_sec=time_limit_sec)
    except Exception:
        pass
    matches = compute_edge_matches(H_sim, V_sim, top_k=top_k)
    right_of = matches["right_of"]; left_of = matches["left_of"]
    bottom_of = matches["bottom_of"]; top_of = matches["top_of"]
    used = set(); grid = [[None]*cols for _ in range(rows)]
    best_sums = [float(np.max(H_sim[i,:]) + np.max(V_sim[i,:]) + np.max(H_sim[:,i]) + np.max(V_sim[:,i])) for i in range(n)]
    start = int(np.argmin(best_sums))
    grid[0][0] = start; used.add(start)
    for r in range(rows):
        for c in range(cols):
            if r==0 and c==0: continue
            candidates = [i for i in range(n) if i not in used]
            viable = []
            for cand in candidates:
                ok = True; score_acc=0.0; cnt=0
                if c>0:
                    left_idx = grid[r][c-1]
                    cond1 = (cand in right_of[left_idx]) or (float(H_sim[left_idx, cand]) >= min_edge_score)
                    cond2 = (left_idx in left_of[cand]) or (float(H_sim[left_idx, cand]) >= min_edge_score)
                    if not (cond1 and cond2): ok=False
                    score_acc += float(H_sim[left_idx, cand]); cnt+=1
                if r>0:
                    top_idx = grid[r-1][c]
                    cond1v = (cand in bottom_of[top_idx]) or (float(V_sim[top_idx, cand]) >= min_edge_score)
                    cond2v = (top_idx in top_of[cand]) or (float(V_sim[top_idx, cand]) >= min_edge_score)
                    if not (cond1v and cond2v): ok=False
                    score_acc += float(V_sim[top_idx, cand]); cnt+=1
                avg = (score_acc/cnt) if cnt>0 else float((np.mean(H_sim[cand,:]) + np.mean(V_sim[cand,:]))/2.0)
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
                if float(H_sim[cur, right]) < min_edge_score:
                    weak.append({"pos": f"{r}_{c}", "dir": "right", "score": float(H_sim[cur, right]), "pair":[int(cur), int(right)]})
            if r+1 < rows:
                bottom = grid[r+1][c]
                if float(V_sim[cur, bottom]) < min_edge_score:
                    weak.append({"pos": f"{r}_{c}", "dir": "bottom", "score": float(V_sim[cur, bottom]), "pair":[int(cur), int(bottom)]})
    placement["weak_edges"] = weak; placement["backtracking_used"] = False
    if use_backtracking and len(weak) > 0:
        try:
            bt = _backtracking_try(H_sim, V_sim, rows, cols, time_limit_sec=time_limit_sec)
            if bt is not None:
                bt["backtracking_used"] = True
                return bt
        except Exception:
            pass
    return placement
