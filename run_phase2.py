#!/usr/bin/env python3
# Run Phase 2 contour-based solver across dataset

import argparse
import os
import traceback
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from phase2.features import load_tiles_from_phase1
from phase2.unified_solver import PuzzleSolver


def parse_args():
    # Parse command line arguments for puzzle solver
    parser = argparse.ArgumentParser(description="Run contour-based puzzle solver on dataset")
    parser.add_argument("--phase1_root", default="phase1_outputs", help="Phase 1 outputs directory")
    parser.add_argument("--out_dir", default="phase2_outputs", help="Output directory")
    parser.add_argument("--group", required=False, help="Puzzle group (e.g., puzzle_2x2). If omitted, runs all groups")
    parser.add_argument("--image", required=False, help="Image ID. If omitted, runs all images")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    parser.add_argument("--dataset_root", default="dataset_images", help="Raw dataset directory (used to run Phase 1 on demand)")
    return parser.parse_args()
from run_phase1 import run_all_phase1


def infer_grid_size(group_name: str):
    # Determine grid dimensions from group name
    if '2x2' in group_name.lower():
        return 2, 2
    elif '4x4' in group_name.lower():
        return 4, 4
    elif '8x8' in group_name.lower():
        return 8, 8
    return None, None


def assemble_puzzle(tiles, placement, rows, cols, output_path):
    # Assemble solved puzzle tiles into final image
    sizes = [(t['img'].shape[0], t['img'].shape[1]) for t in tiles]
    size_counts = {}
    for s in sizes:
        size_counts[s] = size_counts.get(s, 0) + 1
    tile_h, tile_w = max(size_counts.items(), key=lambda kv: kv[1])[0]
    
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            key = f"{r}_{c}"
            if key in placement:
                tile_idx = placement[key]
                if 0 <= tile_idx < len(tiles):
                    tile_img = tiles[tile_idx]['img']
                    h, w = tile_img.shape[:2]
                    
                    if h != tile_h or w != tile_w:
                        tile_img = cv2.resize(tile_img, (tile_w, tile_h))
                    
                    y_start = r * tile_h
                    x_start = c * tile_w
                    canvas[y_start:y_start+tile_h, x_start:x_start+tile_w] = tile_img
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, canvas)
    return output_path


def iter_groups(root_path, specific=None):
    # Iterate over puzzle groups
    if specific:
        yield specific
        return
    if not os.path.isdir(root_path):
        return
    for directory_name in sorted(os.listdir(root_path)):
        full_path = os.path.join(root_path, directory_name)
        if os.path.isdir(full_path) and directory_name.startswith("puzzle_"):
            yield directory_name


def iter_images(root_path, group_name, specific=None):
    # Iterate over images in a group
    group_path = os.path.join(root_path, group_name)
    if specific:
        if os.path.isdir(os.path.join(group_path, specific, "tiles")):
            yield specific
        return
    
    if os.path.isdir(group_path):
        for item in sorted(os.listdir(group_path), key=lambda x: int(x) if x.isdigit() else x):
            item_path = os.path.join(group_path, item)
            if os.path.isdir(item_path):
                tiles_dir = os.path.join(item_path, "tiles")
                if os.path.isdir(tiles_dir):
                    yield item


def process_one(phase1_root, group, image, out_root, time_limit):
    # Process single puzzle: load tiles, solve, and save result
    try:
        rows, cols = infer_grid_size(group)
        if rows is None:
            print(f"[{group}/{image}] SKIP - Cannot infer grid size")
            return False
        
        tiles = load_tiles_from_phase1(phase1_root, group, image)
        print(f"[{group}/{image}] Loaded {len(tiles)} tiles ({rows}x{cols})")
        
        if len(tiles) != rows * cols:
            print(f"[{group}/{image}] WARNING - Expected {rows*cols} tiles, got {len(tiles)}")
        
        solver = PuzzleSolver(tiles, rows, cols)
        result = solver.solve(time_limit=time_limit)
        
        if result is None:
            print(f"[{group}/{image}] FAILED - No solution found")
            return False
        
        placement = result['placement_map']
        score = result.get('score', 0.0)
        method = result.get('method', 'unknown')
        
        output_dir = Path(out_root) / group
        output_dir.mkdir(parents=True, exist_ok=True)

        assembled_path = output_dir / f"{image}.png"
        assemble_puzzle(tiles, placement, rows, cols, str(assembled_path))
        
        print(f"[{group}/{image}] SUCCESS - Score: {score:.3f}, Method: {method}")
        return True
        
    except Exception as e:
        print(f"[{group}/{image}] ERROR - {e}")
        traceback.print_exc()
        return False


def main():
    # Main entry point: process all puzzles or specific ones
    args = parse_args()
    
    print("=" * 70)
    print("Phase 2: Contour-Based Puzzle Solver")
    print("=" * 70)
    
    tasks = [
        (group, image)
        for group in iter_groups(args.phase1_root, args.group)
        for image in iter_images(args.phase1_root, group, args.image)
    ]
    if not tasks:
        print("[INFO] Phase 1 outputs not found for requested puzzles; running Phase 1 now...")
        run_all_phase1(Path(args.dataset_root), Path(args.phase1_root))
        tasks = [
            (group, image)
            for group in iter_groups(args.phase1_root, args.group)
            for image in iter_images(args.phase1_root, group, args.image)
        ]
    if not tasks:
        print("[WARN] No puzzles found to process.")
        return

    max_workers = max(1, os.cpu_count() or 1)
    # Reserve CPUs if launched from run_all
    if os.environ.get("RUN_ALL_CONTEXT"):
        max_workers = max(1, max_workers - 2)
    print(f"[INFO] Using {max_workers} workers for Phase 2")

    successes = 0
    failures = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_one, args.phase1_root, group, image, args.out_dir, args.time_limit): (group, image)
            for (group, image) in tasks
        }

        for fut in as_completed(futures):
            group, image = futures[fut]
            try:
                success = fut.result()
            except Exception as exc:
                print(f"[{group}/{image}] ERROR - {exc}")
                traceback.print_exc()
                failures += 1
                continue

            if success:
                successes += 1
            else:
                failures += 1
    
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total: {successes + failures} | Success: {successes} | Failures: {failures}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
