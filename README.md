# Image Puzzle Project

Lightweight two-phase pipeline for cutting source images into puzzle tiles (Phase 1) and reassembling them with a contour-based solver (Phase 2).

## Setup
1) Install Python 3.10+.
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Phase 1: Preprocess / Tile Cutting
Cuts images into deterministic grids (2x2, 4x4, 8x8 inferred from folder names) and lightly enhances tiles.

Example:
```bash
python run_phase1.py \
  --dataset_root dataset_images \
  --output_root phase1_outputs \
  --max_workers 8 \
  --report_path phase1_outputs/run_report.json \
  --log_level INFO
```

Key flags:
- `--dataset_root`: input dataset root (default: dataset_images).
- `--output_root`: destination for tiles (default: phase1_outputs).
- `--max_workers`: cap CPU processes (default: cpu count).
- `--report_path`: optional JSON summary of processed images.
- `--log_level`: DEBUG/INFO/WARNING/ERROR.

Outputs per image go to `phase1_outputs/<group>/<image_id>/tiles/` with `metadata.json`.

## Phase 2: Solver
Loads Phase 1 tiles, estimates placement with a best-buddies solver, and writes assembled images plus reports.

Example:
```bash
python run_phase2.py \
  --phase1_root phase1_outputs \
  --out_dir phase2_outputs \
  --group puzzle_4x4 \
  --time_limit 60 \
  --max_workers 8 \
  --strip_width 1 \
  --seeds 5 \
  --shifter_iters 8 \
  --seed 1234 \
  --log_level INFO
```

Key flags:
- `--phase1_root`: Phase 1 output root.
- `--out_dir`: Phase 2 outputs.
- `--group` / `--image`: narrow to a specific group or image id.
- `--time_limit`: per-puzzle time budget (seconds).
- `--max_workers`: cap CPU processes.
- `--strip_width`: border strip width used in compatibility extraction; if <1, treated as a fraction of tile min side (default 0.05 → 5% of min side); if >=1, treated as pixels.
- `--seeds`: number of random seeds for placement search (default auto: 2x2->1, 4x4->5, 8x8->10).
- `--shifter_iters`: refinement iterations.
- `--seed`: deterministic RNG seed for reproducible placements.
- `--log_level`: DEBUG/INFO/WARNING/ERROR.

Outputs per puzzle: `phase2_outputs/<group>/<image_id>.png`

## Testing
Run the quick regression suite:
```bash
pytest -q
```

## Notes
- Grid inference expects folder names containing `2x2`, `4x4`, or `8x8`.
- For best results, ensure Phase 1 tiles and masks are generated with the provided preprocessing pipeline.
