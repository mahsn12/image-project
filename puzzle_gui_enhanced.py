#!/usr/bin/env python3
"""
Enhanced Puzzle Solver GUI with smart iteration and dynamic loading.
Iterates through puzzles in order: image 0 (2x2, 4x4, 8x8), image 1 (2x2, 4x4, 8x8), etc.
Shows "Loading..." for images that don't exist yet.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple
import threading
import traceback

from phase2.features import load_tiles_from_phase1
from phase2.unified_solver import PuzzleSolver


class EnhancedPuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle Solver - Visual Inspector")
        self.root.geometry("1400x900")
        
        # Data
        self.puzzles = []  # List of (group, image_id) tuples in smart order
        self.current_idx = 0
        self.phase1_root = "phase1_outputs"
        self.out_dir = "phase2_outputs"
        self.groups_order = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]
        
        # State
        self.solving = False
        self.current_tiles = None
        self.current_result = None
        
        self.setup_ui()
        self.scan_puzzles()
        
    def setup_ui(self):
        """Create the main UI layout."""
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Refresh Puzzles", command=self.scan_puzzles).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Jump-to controls (top-right area)
        self.counter_label = ttk.Label(control_frame, text="0/0")
        self.counter_label.pack(side=tk.RIGHT, padx=5)

        ttk.Button(control_frame, text="Go", command=self.goto_puzzle).pack(side=tk.RIGHT, padx=5)

        self.image_var = tk.StringVar(value="0")
        image_entry = ttk.Entry(control_frame, textvariable=self.image_var, width=6)
        image_entry.pack(side=tk.RIGHT, padx=5)
        ttk.Label(control_frame, text="Image ID:").pack(side=tk.RIGHT)

        self.grid_var = tk.StringVar(value=self.groups_order[0])
        grid_box = ttk.Combobox(control_frame, textvariable=self.grid_var, values=self.groups_order, width=12, state="readonly")
        grid_box.pack(side=tk.RIGHT, padx=5)
        ttk.Label(control_frame, text="Grid:").pack(side=tk.RIGHT)
        
        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Before (Original Tiles)
        left_frame = ttk.LabelFrame(content_frame, text="Before: Original Tiles", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_before = tk.Canvas(left_frame, bg='gray20', width=500, height=500)
        self.canvas_before.pack(fill=tk.BOTH, expand=True)
        
        # Right side - After (Solved Puzzle)
        right_frame = ttk.LabelFrame(content_frame, text="After: Solved Puzzle", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_after = tk.Canvas(right_frame, bg='gray20', width=500, height=500)
        self.canvas_after.pack(fill=tk.BOTH, expand=True)
        
        # Bottom info panel
        info_frame = ttk.LabelFrame(self.root, text="Puzzle Info", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=3, width=100)
        self.info_text.pack(fill=tk.BOTH, expand=False)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        
        ttk.Button(nav_frame, text="< Previous", command=self.previous_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next >", command=self.next_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Resolve (redo)", command=self.resolve_current).pack(side=tk.LEFT, padx=5)
        
    def scan_puzzles(self):
        """
        Scan for available puzzles and organize them in smart order:
        Image 0 (2x2, 4x4, 8x8), Image 1 (2x2, 4x4, 8x8), etc.
        """
        self.puzzles = []
        
        if not os.path.isdir(self.phase1_root):
            self.update_status(f"Phase 1 outputs not found: {self.phase1_root}")
            return
        
        # First, collect all available images
        images_by_group = {}
        
        for group_dir in sorted(os.listdir(self.phase1_root)):
            group_path = os.path.join(self.phase1_root, group_dir)
            if not os.path.isdir(group_path):
                continue
            
            images_by_group[group_dir] = []
            
            for item in sorted(os.listdir(group_path), key=lambda x: int(x) if x.isdigit() else x):
                item_path = os.path.join(group_path, item)
                tiles_dir = os.path.join(item_path, "tiles")
                if os.path.isdir(tiles_dir):
                    images_by_group[group_dir].append(item)
        
        # Now organize by image number first, then by group
        groups_order = self.groups_order
        
        # Find max image ID
        max_image_id = 0
        for images in images_by_group.values():
            for img_id in images:
                if img_id.isdigit():
                    max_image_id = max(max_image_id, int(img_id))
        
        # Build puzzle list in desired order
        for image_id in range(max_image_id + 1):
            for group in groups_order:
                if group in images_by_group:
                    if str(image_id) in images_by_group[group]:
                        self.puzzles.append((group, str(image_id)))
        
        if self.puzzles:
            self.current_idx = 0
            self.update_status(f"Found {len(self.puzzles)} puzzles")
            self.display_puzzle()
        else:
            self.update_status("No puzzles found")
    
    def display_puzzle(self):
        """Display the current puzzle (before and after)."""
        if not self.puzzles or self.current_idx >= len(self.puzzles):
            return
        
        group, image_id = self.puzzles[self.current_idx]
        self.update_status(f"Loading {group}/{image_id}...")
        self.root.update()
        
        try:
            # Try to load tiles
            try:
                self.current_tiles = load_tiles_from_phase1(self.phase1_root, group, image_id)
                self.show_before_image()
            except:
                # If Phase 1 hasn't run yet, show loading
                self.show_loading_image(self.canvas_before, "Loading tiles...")
                self.current_tiles = None
            
            # Try to load solved image
            self.show_after_image(group, image_id)
            
            # Update info
            self.update_info(group, image_id)
            self.update_counter()
            self.update_status(f"Puzzle {group}/{image_id} loaded")
            
        except Exception as e:
            self.update_status(f"Error: {e}")
            self.canvas_before.delete("all")
            self.canvas_after.delete("all")
    
    def show_before_image(self):
        """Display original image from dataset as 'before' image."""
        group, image_id = self.puzzles[self.current_idx]
        
        # Map group name to dataset folder
        if '2x2' in group:
            dataset_folder = "puzzle_2x2"
        elif '4x4' in group:
            dataset_folder = "puzzle_4x4"
        elif '8x8' in group:
            dataset_folder = "puzzle_8x8"
        else:
            self.show_loading_image(self.canvas_before, "Unknown puzzle type")
            return
        
        # Try to load original image from dataset
        original_path = os.path.join("dataset_images", dataset_folder, f"{image_id}.jpg")
        
        if os.path.exists(original_path):
            img = cv2.imread(original_path)
            if img is not None:
                self.show_image_on_canvas(self.canvas_before, img)
                return
        
        # If original image not found, show loading
        self.show_loading_image(self.canvas_before, "Loading original image...")
    
    def show_after_image(self, group: str, image_id: str):
        """Display solved image as 'after' image."""
        solved_path = os.path.join(self.out_dir, group, f"{image_id}.png")
        
        if os.path.exists(solved_path):
            img = cv2.imread(solved_path)
            if img is not None:
                self.show_image_on_canvas(self.canvas_after, img)
                self.current_result = img
                return
        
        # No solved image yet, show loading placeholder
        self.show_loading_image(self.canvas_after, "Loading solution...")
        self.current_result = None
    
    def show_loading_image(self, canvas, text: str):
        """Show a loading message on canvas."""
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1:
            w = 600
        if h <= 1:
            h = 600
        
        blank = np.ones((h, w, 3), dtype=np.uint8) * 200
        cv2.putText(blank, text, (w//2-150, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
        cv2.putText(blank, "(will update automatically)", (w//2-200, h//2+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
        self.show_image_on_canvas(canvas, blank)
    
    def show_image_on_canvas(self, canvas, cv_image):
        """Display OpenCV image on tkinter canvas."""
        # Resize to fit canvas
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w <= 1:
            canvas_w = 600
        if canvas_h <= 1:
            canvas_h = 600
        
        h, w = cv_image.shape[:2]
        # Fill the box almost fully; allow gentle upscaling
        scale = min(canvas_w / w, canvas_h / h) * 0.98
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        img_resized = cv2.resize(cv_image, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo)
        canvas.image = photo
    
    def update_info(self, group: str, image_id: str):
        """Update puzzle information display."""
        self.info_text.delete(1.0, tk.END)
        
        # Determine grid size
        if '2x2' in group:
            grid_n = 2
        elif '4x4' in group:
            grid_n = 4
        elif '8x8' in group:
            grid_n = 8
        else:
            grid_n = "?"
        
        info = f"Group: {group} | Image: {image_id} | Grid: {grid_n}x{grid_n}\n"
        
        if self.current_tiles:
            n_tiles = len(self.current_tiles)
            tile_h, tile_w = self.current_tiles[0]['img'].shape[:2]
            info += f"Tiles: {n_tiles} | Tile size: {tile_w}x{tile_h}\n"
        else:
            info += f"Tiles: Loading...\n"
        
        if self.current_result is not None:
            info += f"Status: ✓ SOLVED\n"
        else:
            solved_path = os.path.join(self.out_dir, group, f"{image_id}.png")
            if os.path.exists(solved_path):
                info += f"Status: ✓ SOLVED\n"
            else:
                info += f"Status: Waiting to be solved...\n"
        
        self.info_text.insert(tk.END, info)
    
    def update_counter(self):
        """Update puzzle counter."""
        if self.puzzles:
            self.counter_label.config(
                text=f"{self.current_idx + 1}/{len(self.puzzles)}"
            )
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def solve_current(self):
        """Solve the current puzzle in a separate thread."""
        if self.solving:
            self.update_status("Already solving...")
            return
        
        if not self.current_tiles:
            self.update_status("Tiles not loaded yet, please wait...")
            return
        
        # Start solving in background thread
        thread = threading.Thread(target=self._solve_worker)
        thread.daemon = True
        thread.start()

    def goto_puzzle(self):
        """Jump to a specific puzzle by grid and image id."""
        target_group = self.grid_var.get()
        target_image = self.image_var.get().strip()
        if not target_image:
            self.update_status("Enter an image id")
            return
        if (target_group, target_image) in self.puzzles:
            self.current_idx = self.puzzles.index((target_group, target_image))
            self.display_puzzle()
            return
        self.update_status(f"Not found: {target_group}/{target_image}")
    
    def _solve_worker(self):
        """Worker thread for solving puzzle."""
        try:
            self.solving = True
            group, image_id = self.puzzles[self.current_idx]
            self.update_status(f"Solving {group}/{image_id}...")
            
            # Determine grid size
            if '2x2' in group:
                grid_n = 2
            elif '4x4' in group:
                grid_n = 4
            elif '8x8' in group:
                grid_n = 8
            else:
                grid_n = int(np.sqrt(len(self.current_tiles)))
            
            # Solve
            tile_imgs = [t['img'] for t in self.current_tiles]
            solver = PuzzleSolver(self.current_tiles, grid_n, grid_n)
            result = solver.solve(time_limit=120.0)
            
            if result is None:
                self.update_status(f"Failed to solve {group}/{image_id}")
                self.solving = False
                return
            
            # Save result
            placement = result['placement_map']
            
            # Assemble image
            tile_h, tile_w = tile_imgs[0].shape[:2]
            canvas = np.zeros((grid_n * tile_h, grid_n * tile_w, 3), dtype=np.uint8)
            
            for r in range(grid_n):
                for c in range(grid_n):
                    key = f"{r}_{c}"
                    if key in placement:
                        tile_idx = placement[key]
                        if 0 <= tile_idx < len(self.current_tiles):
                            tile_img = self.current_tiles[tile_idx]['img']
                            h, w = tile_img.shape[:2]
                            
                            if h != tile_h or w != tile_w:
                                tile_img = cv2.resize(tile_img, (tile_w, tile_h))
                            
                            y_start = r * tile_h
                            x_start = c * tile_w
                            canvas[y_start:y_start+tile_h, x_start:x_start+tile_w] = tile_img
            
            # Save assembled image
            out_path = os.path.join(self.out_dir, group)
            os.makedirs(out_path, exist_ok=True)
            assembled_path = os.path.join(out_path, f"{image_id}.png")
            cv2.imwrite(assembled_path, canvas)
            
            # Store result
            self.current_result = canvas
            
            score = result.get('score', 0.0)
            method = result.get('method', 'unknown')
            
            self.update_status(f"Solved! Score: {score:.3f} ({method})")
            self.update_info(group, image_id)
            
            # Update display
            self.root.after(0, lambda: self.show_after_image(group, image_id))
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.solving = False
    
    def resolve_current(self):
        """Delete current solved image and re-run solver for this puzzle."""
        if not self.puzzles:
            return
        if self.solving:
            self.update_status("Already solving...")
            return
        group, image_id = self.puzzles[self.current_idx]
        solved_path = os.path.join(self.out_dir, group, f"{image_id}.png")
        try:
            if os.path.exists(solved_path):
                os.remove(solved_path)
        except Exception as e:
            self.update_status(f"Could not delete old result: {e}")
        # reset display state
        self.current_result = None
        self.show_loading_image(self.canvas_after, "Re-solving...")
        self.solve_current()
    
    def next_puzzle(self):
        """Move to next puzzle."""
        if self.puzzles:
            self.current_idx = (self.current_idx + 1) % len(self.puzzles)
            self.display_puzzle()
    
    def previous_puzzle(self):
        """Move to previous puzzle."""
        if self.puzzles:
            self.current_idx = (self.current_idx - 1) % len(self.puzzles)
            self.display_puzzle()


def main():
    root = tk.Tk()
    app = EnhancedPuzzleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
