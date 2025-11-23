import numpy as np
from scipy.optimize import linear_sum_assignment


def build_cost_matrix(adj_matrix, rows, cols):
    """
    adj_matrix: N x N matrix of pairwise tile adjacency scores
                adj_matrix[i][j] = similarity score between tiles i and j

    rows, cols: puzzle grid size

    Returns:
        cost_matrix: N x N cost matrix for Hungarian algorithm
    """

    N = rows * cols
    cost_matrix = np.zeros((N, N), dtype=np.float32)

    # max_score used to invert similarity → cost
    max_score = adj_matrix.max()

    # For each tile i
    for tile in range(N):
        for pos in range(N):
            r = pos // cols
            c = pos % cols

            # neighbors expected at this position
            expected = []

            if r > 0:
                expected.append(((r - 1) * cols + c))  # tile above
            if r < rows - 1:
                expected.append(((r + 1) * cols + c))  # tile below
            if c > 0:
                expected.append((r * cols + (c - 1)))  # tile left
            if c < cols - 1:
                expected.append((r * cols + (c + 1)))  # tile right

            if expected:
                # cost = difference between this tile and ideal neighbors
                score = sum(adj_matrix[tile][nbr] for nbr in expected) / len(expected)
            else:
                score = 0

            cost_matrix[tile, pos] = max_score - score

    return cost_matrix


def solve_hungarian(adj_matrix, rows, cols):
    """
    Returns:
        assignment: {tile_index: (row, col)}
    """

    N = rows * cols
    cost_matrix = build_cost_matrix(adj_matrix, rows, cols)

    # Solve assignment
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    layout = {}
    for tile, pos in zip(row_idx, col_idx):
        r = pos // cols
        c = pos % cols
        layout[tile] = (r, c)

    return layout
