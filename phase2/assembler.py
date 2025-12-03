# phase2/assembler.py
import os
import json
import cv2
import numpy as np
from typing import List, Dict
from phase2.solver import save_heatmap

def assemble_and_save(tile_collection: List[Dict], placement_result: Dict, rows: int, cols: int, output_directory: str,
                      horizontal_similarity=None, vertical_similarity=None, global_similarity=None):
    os.makedirs(output_directory, exist_ok=True)
    grid = placement_result["grid"]
    first_tile = tile_collection[0]["img"]
    tile_height, tile_width = first_tile.shape[:2]
    has_alpha_channel = any(t["img"].ndim==3 and t["img"].shape[2]==4 for t in tile_collection)
    assembled_canvas = np.zeros((tile_height*rows, tile_width*cols, 4 if has_alpha_channel else 3), dtype=first_tile.dtype)
    for row_idx in range(rows):
        for col_idx in range(cols):
            tile_index = grid[row_idx][col_idx]
            if tile_index is None: continue
            tile_image = tile_collection[int(tile_index)]["img"]
            y_start = row_idx*tile_height; x_start = col_idx*tile_width
            if assembled_canvas.shape[2]==4 and (tile_image.ndim==3 and tile_image.shape[2]==3):
                alpha_channel = np.ones((tile_image.shape[0], tile_image.shape[1], 1), dtype=tile_image.dtype)*255
                tile_image = np.concatenate([tile_image, alpha_channel], axis=2)
            if assembled_canvas.shape[2]==3 and tile_image.ndim==3 and tile_image.shape[2]==4:
                tile_image = tile_image[:,:,:3]
            assembled_canvas[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_image
    assembled_output_path = os.path.join(output_directory, "assembled.png")
    cv2.imwrite(assembled_output_path, assembled_canvas)
    placement_json_path = os.path.join(output_directory, "placement.json")
    with open(placement_json_path, "w") as output_file:
        json.dump(placement_result.get("placement_map", {}), output_file, indent=2)
    report_json_path = os.path.join(output_directory, "placement_report.json")
    with open(report_json_path, "w") as output_file:
        json.dump({"weak_edges": placement_result.get("weak_edges", []), "backtracking_used": placement_result.get("backtracking_used", False)}, output_file, indent=2)
    if horizontal_similarity is not None:
        save_heatmap(horizontal_similarity, os.path.join(output_directory, "H_similarity_heatmap.png"), "H similarity")
    if vertical_similarity is not None:
        save_heatmap(vertical_similarity, os.path.join(output_directory, "V_similarity_heatmap.png"), "V similarity")
    if global_similarity is not None:
        save_heatmap(global_similarity, os.path.join(output_directory, "global_similarity_heatmap.png"), "Global similarity")
    return {"assembled_path": assembled_output_path, "placement_json": placement_json_path, "report": report_json_path}
