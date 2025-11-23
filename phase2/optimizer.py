import numpy as np
import random
import math


def adjacency_score(layout, adj_matrix, rows, cols):
    """
    Compute total adjacency score for a given layout:
        layout[tile] = (r, c)
    """
    total = 0.0
    N = rows * cols

    # Build reverse lookup: (r,c) -> tile
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for tile, (r, c) in layout.items():
        grid[r][c] = tile

    for r in range(rows):
        for c in range(cols):
            t = grid[r][c]

            # right neighbor
            if c < cols - 1:
                t2 = grid[r][c + 1]
                total += adj_matrix[t, t2]

            # bottom neighbor
            if r < rows - 1:
                t2 = grid[r + 1][c]
                total += adj_matrix[t, t2]

    return total


def swap_positions(layout, t1, t2):
    """
    Swap positions of tiles t1 and t2.
    """
    r1, c1 = layout[t1]
    r2, c2 = layout[t2]
    layout[t1] = (r2, c2)
    layout[t2] = (r1, c1)


def local_hill_climb(layout, adj_matrix, rows, cols, iterations=500):
    """
    Perform greedy local swaps to maximize adjacency score.
    """
    best_score = adjacency_score(layout, adj_matrix, rows, cols)
    tiles = list(layout.keys())

    for _ in range(iterations):
        t1, t2 = random.sample(tiles, 2)
        swap_positions(layout, t1, t2)

        new_score = adjacency_score(layout, adj_matrix, rows, cols)
        if new_score >= best_score:
            best_score = new_score  # keep improvement
        else:
            # revert
            swap_positions(layout, t1, t2)

    return layout


def simulated_annealing(layout, adj_matrix, rows, cols,
                        start_temp=1.0, end_temp=0.001, steps=2000):
    """
    Prevents getting stuck in local maxima.
    """
    tiles = list(layout.keys())
    best_layout = layout.copy()
    best_score = adjacency_score(layout, adj_matrix, rows, cols)

    current_layout = layout.copy()
    current_score = best_score

    for step in range(steps):
        T = start_temp * ((end_temp / start_temp) ** (step / steps))

        t1, t2 = random.sample(tiles, 2)
        swap_positions(current_layout, t1, t2)

        new_score = adjacency_score(current_layout, adj_matrix, rows, cols)
        delta = new_score - current_score

        if delta >= 0 or math.exp(delta / T) > random.random():
            current_score = new_score
        else:
            swap_positions(current_layout, t1, t2)  # reject move

        if current_score > best_score:
            best_score = current_score
            best_layout = current_layout.copy()

    return best_layout


def optimize_layout(initial_layout, adj_matrix, rows, cols):
    """
    Hybrid optimization:
        1. Local hill climbing
        2. Simulated annealing refinement
    """

    # Stage 1: hill climb
    improved = local_hill_climb(initial_layout.copy(),
                                adj_matrix, rows, cols, iterations=800)

    # Stage 2: annealing
    refined = simulated_annealing(improved.copy(),
                                  adj_matrix, rows, cols,
                                  start_temp=1.5, end_temp=0.01, steps=2000)

    return refined
