#!/usr/bin/env python3
"""
Main launcher script for the Puzzle Solver GUI.
Automatically checks and runs Phase 1 and Phase 2 in background while launching GUI.
GUI allows browsing while data is being generated.
"""

import os
import sys
import threading
import subprocess
from pathlib import Path
import time

def check_phase1_complete():
    """Check if Phase 1 has been run."""
    phase1_root = "phase1_outputs"
    if not os.path.isdir(phase1_root):
        return False
    
    # Check if we have at least some outputs
    for item in os.listdir(phase1_root):
        group_path = os.path.join(phase1_root, item)
        if os.path.isdir(group_path):
            for subitem in os.listdir(group_path):
                tiles_dir = os.path.join(group_path, subitem, "tiles")
                if os.path.isdir(tiles_dir):
                    return True  # At least one puzzle processed
    return False


def check_phase2_complete():
    """Check if Phase 2 has been run for at least some puzzles."""
    phase2_root = "phase2_outputs"
    if not os.path.isdir(phase2_root):
        return False
    
    # Check if we have at least some outputs
    for item in os.listdir(phase2_root):
        group_path = os.path.join(phase2_root, item)
        if os.path.isdir(group_path):
            for png_file in os.listdir(group_path):
                if png_file.endswith(".png"):
                    return True  # At least one puzzle solved
    return False


def run_phase1():
    """Run Phase 1 processing."""
    print("\n" + "="*60)
    print("Running Phase 1: Tile Extraction")
    print("="*60)
    try:
        subprocess.run([sys.executable, "run_phase1.py"], check=True)
        print("Phase 1 completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Phase 1 failed with error: {e}")
    except Exception as e:
        print(f"Error running Phase 1: {e}")


def run_phase2():
    """Run Phase 2 solving."""
    print("\n" + "="*60)
    print("Running Phase 2: Puzzle Solving")
    print("="*60)
    try:
        env = os.environ.copy()
        env["RUN_ALL_CONTEXT"] = "1"  # leave two CPUs free while run_all is active
        subprocess.run([sys.executable, "run_phase2.py"], check=True, env=env)
        print("Phase 2 completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Phase 2 failed with error: {e}")
    except Exception as e:
        print(f"Error running Phase 2: {e}")


def run_gui():
    """Launch the GUI."""
    print("\n" + "="*60)
    print("Launching Puzzle Solver GUI")
    print("="*60)
    try:
        subprocess.run([sys.executable, "puzzle_gui_enhanced.py"], check=False)
    except Exception as e:
        print(f"Error launching GUI: {e}")


def main():
    """Main launcher - checks phases and runs GUI with background processing."""
    print("="*60)
    print("Puzzle Solver - Main Launcher")
    print("="*60)
    
    # Check Phase 1
    print("\nChecking Phase 1 status...")
    phase1_done = check_phase1_complete()
    
    if not phase1_done:
        print("Phase 1 outputs not found. Will run Phase 1 in background...")
        phase1_thread = threading.Thread(target=run_phase1, daemon=True)
        phase1_thread.start()
    else:
        print("✓ Phase 1 already completed")
    
    # Check Phase 2
    print("\nChecking Phase 2 status...")
    phase2_done = check_phase2_complete()
    
    if not phase2_done:
        print("Phase 2 outputs not found. Will run Phase 2 in background...")
        # Give Phase 1 a moment to start if needed
        time.sleep(1)
        phase2_thread = threading.Thread(target=run_phase2, daemon=True)
        phase2_thread.start()
    else:
        print("✓ Phase 2 already completed")
    
    # Launch GUI while phases run in background
    print("\nLaunching GUI...")
    print("(Phase 1 and Phase 2 will continue running in background)")
    print("(Images will appear as they become available)")
    run_gui()


if __name__ == "__main__":
    main()
