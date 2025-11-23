import argparse
from pathlib import Path
from .assembler import Assembler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", "-p", required=True, help="Directory inside phase1_outputs")
    ap.add_argument("--out", "-o", default="phase2_outputs", help="Phase 2 output root")
    args = ap.parse_args()

    p1_dir = Path(args.phase1)
    out_root = Path(args.out) / p1_dir.name

    assembler = Assembler(p1_dir)
    assembler.solve(out_root)


if __name__ == "__main__":
    main()
