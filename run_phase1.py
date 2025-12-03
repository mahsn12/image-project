# run_phase1.py  (project root)
from pathlib import Path
import sys
import json

# --- Locate project root and ensure phase1 is importable ---
current_file = Path(__file__).resolve()
project_root = current_file.parent
phase1_directory = project_root / "phase1"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the preprocess functions
from phase1.preprocess import preprocess_image, detect_grid_from_folder

# ---- Hard-coded dataset / output paths (edit if needed) ----
dataset_root_path = project_root / "dataset_images"
output_root_path  = project_root / "phase1_outputs"
# -----------------------------------------------------------

def run_all_phase1(input_dataset_path: Path, output_base_path: Path):
    input_dataset_path = Path(input_dataset_path)
    output_base_path = Path(output_base_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    image_file_list = [file_path for file_path in input_dataset_path.rglob("*") if file_path.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not image_file_list:
        print("[ERROR] No images found in:", input_dataset_path)
        return

    for image_path in image_file_list:
        folder_name = image_path.parent.name
        grid_rows, grid_cols = detect_grid_from_folder(folder_name)
        if grid_rows is None:
            print(f"[SKIP] {image_path}  (folder doesn't contain 2x2/4x4/8x8)")
            continue

        relative_path = image_path.parent.relative_to(input_dataset_path)
        output_directory = output_base_path / relative_path / image_path.stem

        print(f"[PHASE1] Processing: {image_path}")
        output_path, metadata = preprocess_image(image_path, output_directory, grid_rows, grid_cols)

        # tolerant reading of saved-count key (supports both old/new preprocess versions)
        saved_tile_count = metadata.get("num_pieces_saved", None)
        if saved_tile_count is None:
            saved_tile_count = metadata.get("num_tiles_saved", None)

        # if still None, dump meta for debugging
        if saved_tile_count is None:
            print("[WARN] metadata missing expected keys. Full metadata:")
            try:
                print(json.dumps(metadata, indent=2))
            except Exception:
                print(metadata)
            saved_tile_count = "unknown"

        detected_tile_count = metadata.get("num_pieces_detected", metadata.get("num_tiles_detected", "unknown"))

        print(f"[DONE] -> {output_path} | detected={detected_tile_count} saved={saved_tile_count}")

if __name__ == "__main__":
    print("[INFO] Running Phase 1")
    run_all_phase1(dataset_root_path, output_root_path)
    print("[INFO] Phase 1 complete")
