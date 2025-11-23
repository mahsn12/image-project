from pathlib import Path
from phase2.assembler import Assembler

PHASE1_ROOT = "phase1_outputs"
OUTPUT_ROOT = "phase2_outputs"

def run_all_phase2():
    p1_root = Path(PHASE1_ROOT)
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    for puzzle_dir in p1_root.glob("*/*"):
        if not puzzle_dir.is_dir():
            continue

        pre = puzzle_dir / "preprocessed.png"
        meta = puzzle_dir / "metadata.json"
        if not pre.exists() or not meta.exists():
            print(f"[SKIP] {puzzle_dir}")
            continue

        out_dir = out_root / f"{puzzle_dir.parent.name}_{puzzle_dir.name}"
        asm = Assembler(puzzle_dir)
        asm.solve(out_dir)

if __name__ == "__main__":
    run_all_phase2()
