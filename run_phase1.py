# run_phase1.py  (project root)
from pathlib import Path
import sys
import json

# --- Locate project root and ensure phase1 is importable ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
PHASE1_DIR = PROJECT_ROOT / "phase1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the preprocess functions
from phase1.preprocess import preprocess_image, detect_grid_from_folder

# ---- Hard-coded dataset / output paths (edit if needed) ----
DATASET_ROOT = PROJECT_ROOT / "dataset_images"
OUTPUT_ROOT  = PROJECT_ROOT / "phase1_outputs"
# -----------------------------------------------------------

def run_all_phase1(dataset_root: Path, out_root: Path):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in dataset_root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not image_files:
        print("[ERROR] No images found in:", dataset_root)
        return

    for img_path in image_files:
        folder = img_path.parent.name
        rows, cols = detect_grid_from_folder(folder)
        if rows is None:
            print(f"[SKIP] {img_path}  (folder doesn't contain 2x2/4x4/8x8)")
            continue

        rel = img_path.parent.relative_to(dataset_root)
        out_dir = out_root / rel / img_path.stem

        print(f"[PHASE1] Processing: {img_path}")
        out_path, meta = preprocess_image(img_path, out_dir, rows, cols)

        # tolerant reading of saved-count key (supports both old/new preprocess versions)
        saved_count = meta.get("num_pieces_saved", None)
        if saved_count is None:
            saved_count = meta.get("num_tiles_saved", None)

        # if still None, dump meta for debugging
        if saved_count is None:
            print("[WARN] metadata missing expected keys. Full metadata:")
            try:
                print(json.dumps(meta, indent=2))
            except Exception:
                print(meta)
            saved_count = "unknown"

        detected_count = meta.get("num_pieces_detected", meta.get("num_tiles_detected", "unknown"))

        print(f"[DONE] -> {out_path} | detected={detected_count} saved={saved_count}")

if __name__ == "__main__":
    print("[INFO] Running Phase 1")
    run_all_phase1(DATASET_ROOT, OUTPUT_ROOT)
    print("[INFO] Phase 1 complete")
