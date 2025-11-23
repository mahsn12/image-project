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
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_root", default="phase1_outputs", help="Root of phase1 outputs")
    p.add_argument("--out_dir", default="phase2_outputs", help="Root of phase2 outputs")
    p.add_argument("--group", required=False, help="If omitted, run all groups")
    p.add_argument("--image", required=False, help="If omitted, run all images in group")
    p.add_argument("--alpha", type=float, default=0.85, help="Weight for border similarity (0..1)")
    p.add_argument("--gamma", type=float, default=0.0, help="Weight for object similarity (0..1)")
    p.add_argument("--delta", type=float, default=0.12, help="Weight for edge-map similarity (0..1)")
    p.add_argument("--strip_frac", type=float, default=0.10, help="Fraction of tile used for border strips")
    p.add_argument("--top_k", type=int, default=4, help="Top-K edge candidates")
    p.add_argument("--min_edge_score", type=float, default=0.12, help="Min edge score to accept even if not in top-k")
    p.add_argument("--use_backtracking", action="store_true", help="Allow time-limited backtracking to fix weak edges")
    p.add_argument("--obj_edge_frac", type=float, default=0.25, help="Edge-zone fraction for object selection")
    p.add_argument("--obj_alpha_shape", type=float, default=0.6, help="Blend shape vs appearance for object scoring")
    p.add_argument("--use_exact", action="store_true", help="Use exact solver for small puzzles")
    return p.parse_args()

def iter_groups(root, specific=None):
    if specific:
        yield specific; return
    if not os.path.isdir(root):
        raise FileNotFoundError(f"phase1_root not found: {root}")
    for name in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, name)):
            yield name

def iter_images(root, group, specific=None):
    gpath = os.path.join(root, group)
    if specific:
        yield specific; return
    for name in sorted(os.listdir(gpath)):
        if os.path.isdir(os.path.join(gpath, name)):
            yield name

def process_one(phase1_root, group, image, out_root, args):
    out_dir = os.path.join(out_root, group, image)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(phase1_root, group, image, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    rows = int(meta["rows"]); cols = int(meta["cols"])

    tiles = load_tiles_from_phase1(phase1_root, group, image)
    print(f"[{group}/{image}] Loaded {len(tiles)} tiles.")

    H_dist, V_dist = compute_border_distances(tiles, strip_frac=args.strip_frac)
    print(f"[{group}/{image}] Border distances computed.")

    global_sim = compute_global_similarity(tiles)
    print(f"[{group}/{image}] Global similarity computed.")

    # optional object similarity
    object_sim_h = None; object_sim_v = None
    if args.gamma > 0.0:
        try:
            objs = compute_object_descriptors_for_tiles(tiles)
            object_sim_h = compute_object_similarity_matrix(tiles, objs, direction="horizontal", edge_zone_frac=args.obj_edge_frac, alpha_obj_shape=args.obj_alpha_shape)
            object_sim_v = compute_object_similarity_matrix(tiles, objs, direction="vertical", edge_zone_frac=args.obj_edge_frac, alpha_obj_shape=args.obj_alpha_shape)
            print(f"[{group}/{image}] Object similarities computed (gamma={args.gamma}).")
        except Exception as e:
            print(f"[{group}/{image}] WARNING: object-sim failed: {e} -- disabling object-sim for this image.")
            object_sim_h = None; object_sim_v = None

    # edge similarity
    edge_sim_h, edge_sim_v = compute_edge_similarity(tiles, strip_frac=args.strip_frac)
    print(f"[{group}/{image}] Edge (Canny) similarity computed.")

    H_comb, V_comb = build_combined_similarity(
        H_dist, V_dist, global_sim,
        object_sim_h=object_sim_h, object_sim_v=object_sim_v,
        edge_sim_h=edge_sim_h, edge_sim_v=edge_sim_v,
        alpha=args.alpha, gamma=args.gamma, delta=args.delta
    )
    print(f"[{group}/{image}] Combined similarity built (alpha={args.alpha}, gamma={args.gamma}, delta={args.delta}).")

    # save heatmaps
    try:
        save_heatmap(H_comb, os.path.join(out_dir, "H_similarity_heatmap.png"), "H similarity (combined)")
        save_heatmap(V_comb, os.path.join(out_dir, "V_similarity_heatmap.png"), "V similarity (combined)")
        save_heatmap(global_sim, os.path.join(out_dir, "global_similarity_heatmap.png"), "Global similarity")
        save_heatmap(edge_sim_h, os.path.join(out_dir, "edge_similarity_horizontal.png"), "Edge similarity (horizontal)")
        save_heatmap(edge_sim_v, os.path.join(out_dir, "edge_similarity_vertical.png"), "Edge similarity (vertical)")
    except Exception:
        pass

    # choose solver
    if args.use_exact:
        placement = place_tiles_exact(H_comb, V_comb, rows, cols, top_k=args.top_k)
    else:
        placement = place_tiles_with_edge_checks(H_comb, V_comb, rows, cols,
                                                top_k=args.top_k,
                                                min_edge_score=args.min_edge_score,
                                                use_backtracking=args.use_backtracking)
    print(f"[{group}/{image}] Placement done. weak_edges={len(placement.get('weak_edges', []))}, backtracking_used={placement.get('backtracking_used', False)}")

    results = assemble_and_save(tiles, placement, rows, cols, out_dir, H_sim=H_comb, V_sim=V_comb, global_sim=global_sim)
    print(f"[{group}/{image}] Assembled -> {results['assembled_path']}")
    return results

def main():
    args = parse_args()
    failures = {}; total = 0; success = 0
    if args.alpha + args.gamma + args.delta > 1.0 + 1e-12:
        raise ValueError("alpha + gamma + delta must be <= 1.0")
    if not os.path.isdir(args.phase1_root):
        raise FileNotFoundError(args.phase1_root)
    for group in iter_groups(args.phase1_root, args.group):
        for image in iter_images(args.phase1_root, group, args.image):
            total += 1
            key = f"{group}/{image}"
            print("------------------------------------------------------------")
            print("Processing", key)
            try:
                process_one(args.phase1_root, group, image, args.out_dir, args)
                success += 1
            except Exception as e:
                tb = traceback.format_exc()
                failures[key] = {"error": str(e), "traceback": tb}
                print(f"[ERROR] {key}: {e}")
    print("============================================================")
    print(f"Total: {total} | Success: {success} | Failures: {len(failures)}")
    if failures:
        path = os.path.join(args.out_dir, "failures.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(failures, f, indent=2)
        print("Wrote failures to", path)
        raise SystemExit(1)
    raise SystemExit(0)

if __name__ == "__main__":
    main()
