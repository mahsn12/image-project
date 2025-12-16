# Image Puzzle Project

Two-phase pipeline to cut source images into grid tiles (Phase 1) and reassemble them with a mask-free, border-based best-buddies solver (Phase 2). Includes an interactive GUI for viewing before/after results.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Data Layout](#data-layout)
4. [Architecture Overview](#architecture-overview)
5. [Phase 1: Preprocessing & Tile Cutting](#phase-1-preprocessing--tile-cutting)
6. [Phase 2: Puzzle Solving](#phase-2-puzzle-solving)
7. [GUI: Interactive Viewer](#gui-interactive-viewer)
8. [Design Justifications](#design-justifications)
9. [References](#references)
10. [Troubleshooting](#troubleshooting)

## Quick Start

**One command to do everything:**
```bash
python run_all.py
```

This will:
- ✓ Check if Phase 1 has been run (runs it if not)
- ✓ Check if Phase 2 has been run (runs it if not)
- ✓ Launch interactive GUI while processing continues in background
- ✓ Puzzles automatically ordered: image 0 (2x2, 4x4, 8x8), image 1 (2x2, 4x4, 8x8), etc.
- ✓ Shows "Loading..." for images being processed
- ✓ Click Next/Previous to browse while data is being generated

## Setup
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Layout
- Input images live under `dataset_images/`, grouped by folders that include `2x2`, `4x4`, or `8x8` in their name (e.g., `puzzle_4x4/0/*.png`).
- Phase 1 outputs land in `phase1_outputs/<group>/<image_id>/tiles/` with `metadata.json`.
- Phase 2 writes assembled puzzles to `phase2_outputs/<group>/<image_id>.png`.

---

## Architecture Overview

The pipeline consists of two distinct phases:

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT DATASET                        │
│         (Unscrambled source images in grid folders)      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │      PHASE 1: CUTTING        │
    │  • Grid detection            │
    │  • Deterministic tiling      │
    │  • Edge-preserving enhance   │
    └──────────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  PHASE 1 OUTPUTS             │
        │  • tiles/tile_r_c.png        │
        │  • metadata.json             │
        └──────────────────┬───────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   PHASE 2: SOLVING           │
            │  • Border feature extraction │
            │  • Best-buddies matching     │
            │  • Beam-search placement     │
            └──────────────────┬───────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  PHASE 2 OUTPUTS             │
                │  • {image_id}.png (solved)   │
                └──────────────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     GUI VIEWER               │
                    │  • Before/After display      │
                    │  • Interactive navigation    │
                    └──────────────────────────────┘
```

---

## Phase 1: Preprocessing & Tile Cutting

### Purpose
Transform raw puzzle images into standardized, enhanced tiles that preserve edge information for accurate matching.

### Process Flow

#### 1. **Grid Detection**
```python
def detect_grid_from_folder(folder_name: str):
    # Detects grid size (2x2, 4x4, 8x8) from folder name
    # E.g., "puzzle_4x4" → (4, 4)
```

**Justification**: Using folder naming convention for grid specification is deterministic, eliminates need for automatic grid detection (which is ambiguous), and provides user control.

#### 2. **Deterministic Grid Cutting**
Input image → uniformly divided into exact grid cells with no overlapping or gaps.

**Formula:**
```
tile_height = image_height / num_rows
tile_width = image_width / num_cols
tile[r,c] = image[r*H : (r+1)*H, c*W : (c+1)*W]
```

**Justification**: 
- Deterministic approach ensures reproducibility
- No contour detection avoids misalignment issues
- Uniform tiles preserve edge alignment for matching

#### 3. **Edge-Preserving Enhancement Pipeline**
Applied to each tile to enhance matching features while preserving borders:

```
Input Image
    ↓
[1] Bilateral Denoising
    └─ Preserves edges while smoothing textures
    └─ cv2.bilateralFilter(d=9, sigmaColor=40, sigmaSpace=40)
    ↓
[2] CLAHE (Contrast Limited Adaptive Histogram Equalization)
    └─ Enhances local contrast on L channel (LAB color space)
    └─ Prevents edge over-enhancement via clip limit
    ↓
[3] Guided Filtering (or Bilateral Fallback)
    └─ Smooth without blur while preserving edges
    └─ Uses tile itself as guide
    └─ radius=8, eps=1e-2
    ↓
[4] Soft Unsharp Masking
    └─ Crisp edges without halos
    └─ addWeighted(img, 1.12, blur, -0.12)
    ↓
[5] Frequency Fusion
    └─ Blend original (55%) + enhanced (45%)
    └─ Retains detail while keeping enhancement subtle
    ↓
Output: Enhanced Tile
```

**Justification**:
- **Bilateral denoising**: Classic edge-preserving technique from computer vision (Tomasi & Manduchi, 1998)
- **CLAHE**: Prevents over-enhancement at tile edges that could cause false matches
- **Guided filter**: Advanced edge-preserving smoothing (He et al., 2010) better than Gaussian blur
- **Unsharp masking**: Standard technique to enhance local contrast
- **Frequency fusion**: Prevents over-enhancement by blending with original; avoids artificial artifacts that fool edge matchers
- **Adaptive parameters**: Scales with tile size to remain effective across 2x2, 4x4, and 8x8 grids

#### 4. **Metadata Generation**
```json
{
  "grid": [4, 4],
  "tile_height": 256,
  "tile_width": 256,
  "tile_filenames": ["tile_0_0.png", "tile_0_1.png", ...]
}
```

**Justification**: Metadata allows Phase 2 to load tiles in deterministic order, independent of filesystem ordering.

### Output Format
```
phase1_outputs/
├── puzzle_2x2/
│   ├── 0/
│   │   ├── tiles/
│   │   │   ├── tile_0_0.png
│   │   │   ├── tile_0_1.png
│   │   │   ├── tile_1_0.png
│   │   │   └── tile_1_1.png
│   │   └── metadata.json
│   ├── 1/
│   └── ...
├── puzzle_4x4/
└── puzzle_8x8/
```

---

## Phase 2: Puzzle Solving

### Purpose
Reconstruct the original image by analyzing tile borders and finding correct placements using best-buddies matching algorithm.

### Technique: Border-Based Best-Buddies Solver

#### 1. **Mask-Free Border Feature Extraction**
Unlike traditional puzzle solvers that use contour segmentation masks, this approach directly analyzes pixel borders:

**For each tile, extract 4 directional border patches (0=top, 1=right, 2=bottom, 3=left):**
```
       Border 0 (Top)
    ┌──────────────────┐
    │░░░░░░░░░░░░░░░░░│  strip_width = 1 pixel
    │                  │
  B3│     TILE         │B1 (Right)
    │                  │
    │░░░░░░░░░░░░░░░░░│
    └──────────────────┘
       Border 2 (Bottom)
```

**Feature vector for each border includes:**
1. **LAB Color** (3 channels): Separates luminance (L) from color (A, B)
2. **Gradient Magnitude** (1 channel): Edge strength via Sobel operators
3. **Gradient Direction** (1 channel): Edge orientation
4. **Laplacian** (1 channel): Second-derivative edge detector

```python
feature_vector = [L, A, B, |∇|, θ(∇), ∇²]
```

**Justification**:
- **LAB color**: Perceptually uniform color space; L channel matches human brightness perception
- **Gradient**: Captures edge sharpness and orientation at tile borders
- **Laplacian**: Detects fine texture and edge transitions
- **Multiple scales**: Combines different edge representations for robustness
- **No contours**: Avoids segmentation artifacts; works directly on pixels

#### 2. **Compatibility Matrix Computation**
Build 4 matrices (one per direction) measuring how well adjacent tiles match:

```
For side 0 (top), tile i's top border should match tile j's bottom:
    compat[0][i, j] = distance(border_i[top], border_j[bottom])

Formula with weighted distances:
    D(A, B) = [Σ|A-B|^p]^(q/p)
    
    Where:
    - p = 0.3 (sub-linear norm)
    - q = 1/16 (compression factor)
    - Weighted combination:
        0.4 × color_distance
      + 0.2 × gradient_magnitude_distance
      + 0.2 × gradient_direction_distance
      + 0.4 × laplacian_distance
```

**Justification**:
- **Sublinear norm (p=0.3)**: Makes outliers less influential; fewer false positives
- **Weighted channels**: Laplacian and color weighted equally (0.4) as primary cues; gradient magnitude/direction (0.2) as secondary
- **4 directional matrices**: Separates constraint spaces for directional matching

#### 3. **Best-Buddies Matching**
Find strongly reciprocal matches:

```
Tile A's best match in direction 0 (top) = Tile B
Tile B's best match in direction 2 (bottom) = Tile A
→ Strong confidence that A and B are adjacent
```

**Algorithm:**
1. For each tile and direction, find top-K candidates (candidates below distance threshold)
2. Check bidirectional agreement: A→B in direction 0 AND B→A in direction 2
3. Build confident edges in the placement graph

**Justification**:
- **Bidirectional agreement**: Dramatically reduces false matches vs single-direction voting
- **Beam search**: Explores multiple promising partial solutions instead of greedy left-to-right
- **Dynamic seeding**: Starts with most confident placements, allowing less confident corners to be resolved later

#### 4. **Beam-Search Placement with Dynamic Seeding**
Build the solution incrementally using beam search:

```
Initialization:
  For each possible start tile:
    Place at (0,0)
    Propagate constraints via best-buddies graph
    →State = [partial placement, cost, unplaced tiles]

Iteration:
  For each state in beam (top K by cost):
    Try placing each remaining tile at empty position
    If best-buddies confirm it:
      Add to new beam
  Prune to top K states by cost

Termination:
  When all tiles placed or beam exhausted
  → Return best state found
```

**Dynamic Seeding Benefits:**
- Doesn't assume (0,0) is known
- Explores multiple root placements in parallel
- Adapts to missing edge matches

**Justification**:
- **Beam search**: Explores K promising paths vs greedy single path (more robust to local minima)
- **Best-buddies prioritization**: Heavily weighted constraints reduce search space
- **Cost function balances**:
  - Strongly rewards best-buddy placements
  - Penalizes mismatches
  - Handles missing constraints gracefully

#### 5. **Solution Assembly**
Once placement found:
```
For each position (r,c):
  Retrieve tile at grid[r,c]
  Extract tile image from phase1_outputs
  Place at canvas position (r*tile_h, c*tile_w)
Save as phase2_outputs/{group}/{image_id}.png
```

### Output Format
```
phase2_outputs/
├── puzzle_2x2/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── puzzle_4x4/
└── puzzle_8x8/
```

---

## GUI: Interactive Viewer

### Features
- **Left panel**: Original puzzle image from dataset (unscrambled reference)
- **Right panel**: Solved puzzle image (output from Phase 2)
- **Navigation**: Browse through puzzles while Phase 1/2 processing continues in background
- **Smart ordering**: Puzzles grouped by image ID first, then grid size (compare same image across difficulties)
- **Dynamic loading**: Shows "Loading..." for incomplete data; auto-updates when ready
- **Refresh**: Force re-scan of available puzzles

### Controls
```
[Refresh Puzzles]  [Solve Current]
[< Previous]  [Next >]  [Solve & Next]

Image counter: n/total
Status: Ready/Loading/Solving
```

### Technical Stack
- **tkinter**: GUI framework (built-in, cross-platform)
- **PIL/Pillow**: Image display and resizing
- **OpenCV (cv2)**: Image manipulation
- **Threading**: Non-blocking background processes
- **Subprocess**: Launch Phase 1/2 without blocking GUI

---

## Design Justifications

### Why No Contour Segmentation?
**Traditional approaches** (e.g., Jigsaw solvers using contour masks):
- Require precise contour detection
- Fail on worn edges, curved jigsaw pieces, or puzzle-like images
- Add computational overhead

**This approach** (border pixels directly):
- Works on any image grid (photos, art, documents)
- Robust to worn or curved edges
- Simpler implementation

### Why Best-Buddies Matching?
**Alternatives:**
1. **Greedy left-to-right**: Fast but error-prone; early mistakes cascade
2. **Hungarian algorithm**: Optimal but O(n³) complexity; too slow for 64+ tiles
3. **Genetic algorithms**: Slow convergence

**Best-buddies + beam search:**
- Exploits strong reciprocal constraints (A↔B must match)
- Beam search explores K promising paths (more robust than greedy)
- Near-linear complexity for well-constrained puzzles
- Scales well to 64-tile puzzles (8x8 grid)

### Why LAB Color Space?
- **Separates luminance from color**: Gradient detection more robust
- **Perceptually uniform**: Color differences match human perception
- **Standard in image processing**: Used in SIFT, SURF, and modern CNN backbones

### Why Adaptive Enhancement Parameters?
Large tiles (4x4, 8x8) have different texture scales than small tiles (2x2):
- Bilateral filter radius scales with tile size
- CLAHE grid size adapts to avoid over-segmentation
- Guided filter radius matches tile structure
→ Same code works well across 2x2, 4x4, 8x8 without per-size tuning

### Why Beam Search Breadth?
Default: **K=5** candidate states per iteration
- Explores 5 promising partial solutions in parallel
- Balances exploration vs computation time
- For most 4x4/8x8 puzzles: finds solution in first 1-2 iterations
- Fallback to lower K if time limit exceeded

### Why Frequency Fusion (0.55 orig + 0.45 enhanced)?
- Pure enhanced image: Over-processed; artificial edges fool matcher
- Pure original image: Blurry; loses edge detail
- 55/45 blend: Retains original texture while boosting real edges
→ Best-buddies matcher gets natural, convincing borders

---

## References

### Academic Papers
1. **Best-Buddies Similarity** - Freeman & Garland (2002) "Image Quilting for Texture Synthesis and Transfer"
   - Foundational bidirectional matching concept
   
2. **LAB Color Space** - CIE 1976 Color Space
   - Perceptually uniform opponent color model
   
3. **Bilateral Filtering** - Tomasi & Manduchi (1998) "Bilateral Filtering for Gray and Color Images"
   - Edge-preserving denoising technique
   
4. **Guided Filter** - He, Sun, Tang (2010) "Guided Image Filtering"
   - Advanced edge-preserving smoothing
   
5. **CLAHE** - Pizer et al. (1987) "Adaptive Histogram Equalization and Its Variations"
   - Local contrast enhancement while controlling over-enhancement

### Standard Computer Vision Techniques
- **Sobel/Laplacian operators**: Gradient detection (standard edge detection)
- **Gaussian blur**: Low-pass filtering
- **Unsharp masking**: Local contrast enhancement
- **Image resizing (bilinear interpolation)**: cv2.INTER_LINEAR

### Algorithms
- **Beam search**: Classic search algorithm used in NLP, planning
- **Compatibility matrices**: CSP (Constraint Satisfaction Problem) concept
- **Greedy/approximate matching**: Practical alternative to exhaustive search

---

## Troubleshooting
- "No puzzles found": ensure `dataset_images/` is populated and folders contain `2x2`, `4x4`, or `8x8` in their names. Phase 2 will attempt to generate missing Phase 1 outputs automatically.
- "Cannot read image": verify image paths and permissions under `dataset_images/`.
- **GUI not launching**: Ensure tkinter is installed (built-in on Windows/Mac; on Linux run `sudo apt-get install python3-tk`).

## Testing
```bash
pytest -q
```
