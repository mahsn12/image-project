import os
import json
from pathlib import Path
import cv2

from .utils import apply_clahe, bilateral_smooth, unsharp_mask


def detect_grid_from_folder(folder_name: str):
    name = folder_name.lower()
    for g in ["2x2", "4x4", "8x8"]:
        if g in name:
            r, c = g.split("x")
            return int(r), int(c)
    return None, None


def preprocess_image(in_path: Path, out_path: Path, rows: int, cols: int):
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    # PHASE 1 PIPELINE
    img1 = apply_clahe(img)
    img2 = bilateral_smooth(img1)
    img3 = unsharp_mask(img2)

    out_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path / "preprocessed.png"), img3)

    meta = {
        "source": str(in_path),
        "rows": rows,
        "cols": cols
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_phase1(dataset_root: str, output_root: str = "phase1_outputs"):
    dataset_root = Path(dataset_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images = list(dataset_root.rglob("*.*"))
    images = [x for x in images if x.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    for img_path in images:
        folder = img_path.parent.name
        rows, cols = detect_grid_from_folder(folder)

        if rows is None:
            print(f"[SKIP] {img_path} (folder name does not contain grid size)")
            continue

        rel = img_path.parent.relative_to(dataset_root)
        out_dir = out_root / rel / img_path.stem

        preprocess_image(img_path, out_dir, rows, cols)
        print(f"[OK] {img_path} -> {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", required=True)
    ap.add_argument("--out", "-o", default="phase1_outputs")
    args = ap.parse_args()
    run_phase1(args.dataset, args.out)
