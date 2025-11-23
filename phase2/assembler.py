# phase2/assembler.py
import os
import json
import cv2
import numpy as np
from typing import List, Dict
from phase2.solver import save_heatmap

def assemble_and_save(tiles: List[Dict], placement: Dict, rows: int, cols: int, out_dir: str,
                      H_sim=None, V_sim=None, global_sim=None):
    os.makedirs(out_dir, exist_ok=True)
    grid = placement["grid"]
    first = tiles[0]["img"]
    th, tw = first.shape[:2]
    has_alpha = any(t["img"].ndim==3 and t["img"].shape[2]==4 for t in tiles)
    canvas = np.zeros((th*rows, tw*cols, 4 if has_alpha else 3), dtype=first.dtype)
    for r in range(rows):
        for c in range(cols):
            idx = grid[r][c]
            if idx is None: continue
            tile_img = tiles[int(idx)]["img"]
            y0 = r*th; x0 = c*tw
            if canvas.shape[2]==4 and (tile_img.ndim==3 and tile_img.shape[2]==3):
                alpha = np.ones((tile_img.shape[0], tile_img.shape[1], 1), dtype=tile_img.dtype)*255
                tile_img = np.concatenate([tile_img, alpha], axis=2)
            if canvas.shape[2]==3 and tile_img.ndim==3 and tile_img.shape[2]==4:
                tile_img = tile_img[:,:,:3]
            canvas[y0:y0+th, x0:x0+tw] = tile_img
    assembled_path = os.path.join(out_dir, "assembled.png")
    cv2.imwrite(assembled_path, canvas)
    placement_json = os.path.join(out_dir, "placement.json")
    with open(placement_json, "w") as f:
        json.dump(placement.get("placement_map", {}), f, indent=2)
    report_path = os.path.join(out_dir, "placement_report.json")
    with open(report_path, "w") as f:
        json.dump({"weak_edges": placement.get("weak_edges", []), "backtracking_used": placement.get("backtracking_used", False)}, f, indent=2)
    if H_sim is not None:
        save_heatmap(H_sim, os.path.join(out_dir, "H_similarity_heatmap.png"), "H similarity")
    if V_sim is not None:
        save_heatmap(V_sim, os.path.join(out_dir, "V_similarity_heatmap.png"), "V similarity")
    if global_sim is not None:
        save_heatmap(global_sim, os.path.join(out_dir, "global_similarity_heatmap.png"), "Global similarity")
    return {"assembled_path": assembled_path, "placement_json": placement_json, "report": report_path}
