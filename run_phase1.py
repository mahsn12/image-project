from pathlib import Path
from phase1.preprocess import preprocess_image, detect_grid_from_folder
import cv2
import json

# HARD-CODED PATHS
DATASET_ROOT = "dataset_images"
OUTPUT_ROOT = "phase1_outputs"


def run_all_phase1(dataset_root, out_root):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Find every image file inside dataset
    image_files = [
        p for p in dataset_root.rglob("*")
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    if not image_files:
        print("[ERROR] No images found inside dataset folder:", dataset_root)
        return

    for img_path in image_files:

        # Detect grid size
        folder = img_path.parent.name
        rows, cols = detect_grid_from_folder(folder)
        if rows is None:
            print(f"[SKIP] {img_path} (folder name does not match 2x2/4x4/8x8)")
            continue

        # Determine output path
        rel_path = img_path.parent.relative_to(dataset_root)
        out_dir = out_root / rel_path / img_path.stem

        print(f"\n[GROUP] Processing Phase 1: {img_path}")
        preprocess_image(img_path, out_dir, rows, cols)
        print(f"[DONE] → {out_dir}")


if __name__ == "__main__":
    print(f"[INFO] Running Phase 1 on dataset: {DATASET_ROOT}")
    run_all_phase1(DATASET_ROOT, OUTPUT_ROOT)
    print("[DONE] Phase 1 completed.")
