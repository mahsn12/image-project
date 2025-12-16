# Image Puzzle Project

Two-phase pipeline to cut source images into grid tiles (Phase 1) and reassemble them with a contour-based best-buddies solver (Phase 2).

## Setup
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data layout
- Input images live under `dataset_images/`, grouped by folders that include `2x2`, `4x4`, or `8x8` in their name (e.g., `puzzle_4x4/0/*.png`).
- Phase 1 outputs land in `phase1_outputs/<group>/<image_id>/tiles/` with `metadata.json`.
- Phase 2 writes assembled puzzles to `phase2_outputs/<group>/<image_id>.png`.

## Phase 1: preprocessing / tile cutting
Cuts each image into a deterministic grid (2x2, 4x4, 8x8 inferred from the parent folder name), lightly enhances tiles, and writes the enhanced tiles to disk.

Usage (paths are already set in the script; edit there if needed):
```bash
python run_phase1.py
```

Outputs per image:
- `tiles/tile_rr_cc.png` (enhanced tile)
- `metadata.json` (grid, tile sizes, filenames)

## Phase 2: solver (auto-runs Phase 1 if needed)
Loads Phase 1 tiles, estimates placement with a best-buddies solver, and saves the assembled image. If Phase 1 outputs are missing for the requested puzzles, Phase 2 will first run Phase 1 using `--dataset_root` and then continue.

Common invocations:
```bash
# Run everything (will trigger Phase 1 if outputs are absent)
python run_phase2.py

# Run a specific group
python run_phase2.py --group puzzle_4x4

# Run a single puzzle within a group
python run_phase2.py --group puzzle_4x4 --image 5
```

Key flags:
- `--phase1_root` (default `phase1_outputs`): where Phase 1 tiles are read from.
- `--dataset_root` (default `dataset_images`): raw dataset; used if Phase 1 needs to be run on demand.
- `--out_dir` (default `phase2_outputs`): assembled outputs.
- `--group` / `--image`: narrow execution to a specific folder or image ID.
- `--time_limit`: per-puzzle solve budget in seconds.

## Troubleshooting
- "No puzzles found": ensure `dataset_images/` is populated and folders contain `2x2`, `4x4`, or `8x8` in their names. Phase 2 will attempt to generate missing Phase 1 outputs automatically.
- "Cannot read image": verify image paths and permissions under `dataset_images/`.

## Testing
```bash
pytest -q
```
