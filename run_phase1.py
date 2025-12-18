# run_phase1.py (project root)
from pathlib import Path
import sys
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup project path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent
phase1_directory = project_root / "phase1"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import preprocessing functions
from phase1.preprocess import preprocess_image, detect_grid_from_folder

# Configure dataset and output paths
dataset_root_path = project_root / "dataset_images"
output_root_path  = project_root / "phase1_outputs"

def _process_one(image_path: Path, input_dataset_path: Path, output_base_path: Path):
    # Process single image: detect grid, extract and enhance tiles
    folder_name = image_path.parent.name
    grid_rows, grid_cols = detect_grid_from_folder(folder_name)
    if grid_rows is None:
        return ("skip", image_path, None, None, None)

    relative_path = image_path.parent.relative_to(input_dataset_path)
    output_directory = output_base_path / relative_path / image_path.stem

    output_path, metadata = preprocess_image(
        image_path,
        output_directory,
        grid_rows,
        grid_cols,
    )

    saved_tile_count = metadata.get("num_pieces_saved", None)
    if saved_tile_count is None:
        saved_tile_count = metadata.get("num_tiles_saved", None)
    if saved_tile_count is None:
        saved_tile_count = "unknown"

    detected_tile_count = metadata.get("num_pieces_detected", metadata.get("num_tiles_detected", "unknown"))
    return ("done", image_path, output_path, detected_tile_count, saved_tile_count)


def run_all_phase1(input_dataset_path: Path, output_base_path: Path):
    # Process all images in dataset with multi-processing
    input_dataset_path = Path(input_dataset_path)
    output_base_path = Path(output_base_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    image_file_list = [file_path for file_path in input_dataset_path.rglob("*") if file_path.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not image_file_list:
        print("[ERROR] No images found in:", input_dataset_path)
        return

    max_workers = max(1, os.cpu_count() or 1)
    print(f"[INFO] Using {max_workers} workers for Phase 1")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_one, image_path, input_dataset_path, output_base_path): image_path
            for image_path in image_file_list
        }

        for fut in as_completed(futures):
            try:
                status, image_path, output_path, detected_tile_count, saved_tile_count = fut.result()
            except Exception as exc:
                image_path = futures[fut]
                print(f"[ERROR] {image_path} failed: {exc}")
                continue

            if status == "skip":
                print(f"[SKIP] {image_path} (folder doesn't contain 2x2/4x4/8x8)")
                continue
            print(f"[DONE] -> {output_path} | detected={detected_tile_count} saved={saved_tile_count}")

if __name__ == "__main__":
    print("[INFO] Running Phase 1")
    run_all_phase1(dataset_root_path, output_root_path)
    print("[INFO] Phase 1 complete")
