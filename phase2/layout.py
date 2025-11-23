import numpy as np
from .hungarian_solver import solve_hungarian
from .optimizer import optimize_layout


def build_adjacency_matrix(tiles_edges):
    """
    tiles_edges: list of dicts, each containing {top, bottom, left, right} edge strips.
                 strips are grayscale arrays of fixed size (e.g., 64x64).

    Returns:
        adj_matrix: N x N matrix of symmetric similarity scores
    """

    N = len(tiles_edges)
    adj_matrix = np.zeros((N, N), dtype=np.float32)

    def match_score(A, B):
        """Symmetric similarity: best match of all directions."""
        # A-bottom  ↔ B-top
        s1 = _ncc(A["bottom"], B["top"])
        # A-top     ↔ B-bottom
        s2 = _ncc(A["top"], B["bottom"])
        # A-right   ↔ B-left
        s3 = _ncc(A["right"], B["left"])
        # A-left    ↔ B-right
        s4 = _ncc(A["left"], B["right"])
        return max(s1, s2, s3, s4)

    # Local NCC helper (same used in your old strip logic)
    def _ncc(a, b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a -= a.mean()
        b -= b.mean()
        denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()))
        if denom < 1e-6:
            return -1
        return float((a*b).sum() / denom)

    # Build symmetric adjacency matrix
    for i in range(N):
        for j in range(N):
            if i != j:
                adj_matrix[i, j] = match_score(tiles_edges[i], tiles_edges[j])

    return adj_matrix


def solve_layout(tiles_edges, rows, cols):
    """
    Master layout solver:
       1) Compute adjacency matrix from feature strips
       2) Global assignment via Hungarian algorithm
       3) Optimize via swap-based refinement

    Returns:
       final_layout: {tile_index: (row, col)}
    """

    # Step 1: adjacency matrix
    adj_matrix = build_adjacency_matrix(tiles_edges)

    # Step 2: Hungarian assignment
    initial_layout = solve_hungarian(adj_matrix, rows, cols)

    # Step 3: refine with optimizer
    final_layout = optimize_layout(initial_layout, adj_matrix, rows, cols)

    return final_layout, adj_matrix
