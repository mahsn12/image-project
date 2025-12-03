# run_phase2.py
"""
Runner for Phase-2. Works over all groups/images by default.
"""

import argparse
import os
import json
import traceback

from phase2.features import (
    load_tiles_from_phase1,
    compute_border_distances,
    compute_global_similarity,
    compute_object_descriptors_for_tiles,
    compute_object_similarity_matrix,
    compute_edge_similarity
)
from phase2.solver import build_combined_similarity, place_tiles_with_edge_checks, place_tiles_exact, save_heatmap
from phase2.assembler import assemble_and_save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_root", default="phase1_outputs", help="Root of phase1 outputs")
    parser.add_argument("--out_dir", default="phase2_outputs", help="Root of phase2 outputs")
    parser.add_argument("--group", required=False, help="If omitted, run all groups")
    parser.add_argument("--image", required=False, help="If omitted, run all images in group")
    parser.add_argument("--alpha", type=float, default=0.85, help="Weight for border similarity (0..1)")
    parser.add_argument("--gamma", type=float, default=0.0, help="Weight for object similarity (0..1)")
    parser.add_argument("--delta", type=float, default=0.12, help="Weight for edge-map similarity (0..1)")
    parser.add_argument("--strip_frac", type=float, default=0.10, help="Fraction of tile used for border strips")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K edge candidates")
    parser.add_argument("--min_edge_score", type=float, default=0.12, help="Min edge score to accept even if not in top-k")
    parser.add_argument("--use_backtracking", action="store_true", help="Allow time-limited backtracking to fix weak edges")
    parser.add_argument("--obj_edge_frac", type=float, default=0.25, help="Edge-zone fraction for object selection")
    parser.add_argument("--obj_alpha_shape", type=float, default=0.6, help="Blend shape vs appearance for object scoring")
    parser.add_argument("--use_exact", action="store_true", help="Use exact solver for small puzzles")
    return parser.parse_args()

def iter_groups(root_path, specific=None):
    if specific:
        yield specific; return
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"phase1_root not found: {root_path}")
    for directory_name in sorted(os.listdir(root_path)):
        if os.path.isdir(os.path.join(root_path, directory_name)):
            yield directory_name

def iter_images(root_path, group_name, specific=None):
    group_path = os.path.join(root_path, group_name)
    if specific:
        yield specific; return
    for directory_name in sorted(os.listdir(group_path)):
        if os.path.isdir(os.path.join(group_path, directory_name)):
            yield directory_name

def process_one(phase1_root, group, image, out_root, configuration):
    output_directory = os.path.join(out_root, group, image)
    os.makedirs(output_directory, exist_ok=True)
    metadata_path = os.path.join(phase1_root, group, image, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)
    with open(metadata_path, "r") as file_handle:
        metadata = json.load(file_handle)
    rows = int(metadata["rows"]); cols = int(metadata["cols"])

    tile_collection = load_tiles_from_phase1(phase1_root, group, image)
    print(f"[{group}/{image}] Loaded {len(tile_collection)} tiles.")

    horizontal_distance, vertical_distance = compute_border_distances(tile_collection, strip_frac=configuration.strip_frac)
    print(f"[{group}/{image}] Border distances computed.")

    global_similarity = compute_global_similarity(tile_collection)
    print(f"[{group}/{image}] Global similarity computed.")

    # optional object similarity
    object_similarity_h = None; object_similarity_v = None
    if configuration.gamma > 0.0:
        try:
            objects = compute_object_descriptors_for_tiles(tile_collection)
            object_similarity_h = compute_object_similarity_matrix(tile_collection, objects, direction="horizontal", edge_zone_frac=configuration.obj_edge_frac, shape_weight=configuration.obj_alpha_shape)
            object_similarity_v = compute_object_similarity_matrix(tile_collection, objects, direction="vertical", edge_zone_frac=configuration.obj_edge_frac, shape_weight=configuration.obj_alpha_shape)
            print(f"[{group}/{image}] Object similarities computed (gamma={configuration.gamma}).")
        except Exception as exception_obj:
            print(f"[{group}/{image}] WARNING: object-sim failed: {exception_obj} -- disabling object-sim for this image.")
            object_similarity_h = None; object_similarity_v = None

    # edge similarity
    edge_similarity_h, edge_similarity_v = compute_edge_similarity(tile_collection, fraction=configuration.strip_frac)
    print(f"[{group}/{image}] Edge (Canny) similarity computed.")

    horizontal_combined, vertical_combined = build_combined_similarity(
        horizontal_distance, vertical_distance, global_similarity,
        object_similarity_h=object_similarity_h, object_similarity_v=object_similarity_v,
        edge_similarity_h=edge_similarity_h, edge_similarity_v=edge_similarity_v,
        alpha=configuration.alpha, gamma=configuration.gamma, delta=configuration.delta
    )
    print(f"[{group}/{image}] Combined similarity built (alpha={configuration.alpha}, gamma={configuration.gamma}, delta={configuration.delta}).")

    # save heatmaps
    try:
        save_heatmap(horizontal_combined, os.path.join(output_directory, "H_similarity_heatmap.png"), "H similarity (combined)")
        save_heatmap(vertical_combined, os.path.join(output_directory, "V_similarity_heatmap.png"), "V similarity (combined)")
        save_heatmap(global_similarity, os.path.join(output_directory, "global_similarity_heatmap.png"), "Global similarity")
        save_heatmap(edge_similarity_h, os.path.join(output_directory, "edge_similarity_horizontal.png"), "Edge similarity (horizontal)")
        save_heatmap(edge_similarity_v, os.path.join(output_directory, "edge_similarity_vertical.png"), "Edge similarity (vertical)")
    except Exception:
        pass

    # choose solver
    if configuration.use_exact:
        placement = place_tiles_exact(horizontal_combined, vertical_combined, rows, cols, top_k=configuration.top_k)
    else:
        placement = place_tiles_with_edge_checks(horizontal_combined, vertical_combined, rows, cols,
                                                top_k=configuration.top_k,
                                                min_edge_score=configuration.min_edge_score,
                                                use_backtracking=configuration.use_backtracking)
    print(f"[{group}/{image}] Placement done. weak_edges={len(placement.get('weak_edges', []))}, backtracking_used={placement.get('backtracking_used', False)}")

    results = assemble_and_save(tile_collection, placement, rows, cols, output_directory, horizontal_similarity=horizontal_combined, vertical_similarity=vertical_combined, global_similarity=global_similarity)
    print(f"[{group}/{image}] Assembled -> {results['assembled_path']}")
    return results

def main():
    configuration = parse_args()
    failure_log = {}; total_count = 0; success_count = 0
    if configuration.alpha + configuration.gamma + configuration.delta > 1.0 + 1e-12:
        raise ValueError("alpha + gamma + delta must be <= 1.0")
    if not os.path.isdir(configuration.phase1_root):
        raise FileNotFoundError(configuration.phase1_root)
    for group in iter_groups(configuration.phase1_root, configuration.group):
        for image in iter_images(configuration.phase1_root, group, configuration.image):
            total_count += 1
            result_key = f"{group}/{image}"
            print("------------------------------------------------------------")
            print("Processing", result_key)
            try:
                process_one(configuration.phase1_root, group, image, configuration.out_dir, configuration)
                success_count += 1
            except Exception as exception_obj:
                traceback_str = traceback.format_exc()
                failure_log[result_key] = {"error": str(exception_obj), "traceback": traceback_str}
                print(f"[ERROR] {result_key}: {exception_obj}")
    print("============================================================")
    print(f"Total: {total_count} | Success: {success_count} | Failures: {len(failure_log)}")
    if failure_log:
        failure_path = os.path.join(configuration.out_dir, "failures.json")
        os.makedirs(configuration.out_dir, exist_ok=True)
        with open(failure_path, "w") as output_file:
            json.dump(failure_log, output_file, indent=2)
        print("Wrote failures to", failure_path)
        raise SystemExit(1)
    raise SystemExit(0)

if __name__ == "__main__":
    main()
