import cv2
import json
import numpy as np
from pathlib import Path

from .splitter import split_into_tiles
from .layout import solve_layout
from .features import extract_edge_strips


class Assembler:
    def __init__(self, phase1_dir: Path):
        self.phase1_dir = phase1_dir

        self.img_path = phase1_dir / "preprocessed.png"
        self.meta_path = phase1_dir / "metadata.json"

        if not self.img_path.exists():
            raise RuntimeError(f"Missing preprocessed image: {self.img_path}")
        if not self.meta_path.exists():
            raise RuntimeError(f"Missing metadata: {self.meta_path}")

        with open(self.meta_path) as f:
            meta = json.load(f)

        self.rows = meta["rows"]
        self.cols = meta["cols"]

        self.img = cv2.imread(str(self.img_path))
        if self.img is None:
            raise RuntimeError(f"Cannot read image: {self.img_path}")

    def solve(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # 1. Split into tiles
        # -----------------------------
        tiles, coords = split_into_tiles(self.img, self.rows, self.cols)

        # -----------------------------
        # 2. Extract edge strips
        # -----------------------------
        tiles_edges = [extract_edge_strips(t) for t in tiles]

        # -----------------------------
        # 3. Solve layout
        # -----------------------------
        layout, adj_matrix = solve_layout(tiles_edges, self.rows, self.cols)

        # -----------------------------
        # 4. Assemble final image
        # -----------------------------
        tile_h = tiles[0].shape[0]
        tile_w = tiles[0].shape[1]

        final = np.zeros((self.rows * tile_h,
                          self.cols * tile_w, 3), dtype=np.uint8)

        for tile_idx, (r, c) in layout.items():
            final[
                r * tile_h:(r + 1) * tile_h,
                c * tile_w:(c + 1) * tile_w
            ] = tiles[tile_idx]

        # -----------------------------
        # 5. Save output
        # -----------------------------
        cv2.imwrite(str(out_dir / "assembled.png"), final)

        with open(out_dir / "layout.json", "w") as f:
            json.dump({int(k): [int(v[0]), int(v[1])] 
                       for k, v in layout.items()}, f, indent=2)

        # Save adjacency matrix for debugging
        np.save(str(out_dir / "adj_matrix.npy"), adj_matrix)

        # Save tiles
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        for idx, tile in enumerate(tiles):
            cv2.imwrite(str(tiles_dir / f"tile_{idx}.png"), tile)

        print(f"[OK] Assembled puzzle saved in: {out_dir}")
