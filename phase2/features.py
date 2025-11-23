import cv2
import numpy as np


def extract_edge_strips(tile, strip_frac=0.15):
    """
    Extracts top, bottom, left, and right edge strips from a tile.
    Strips are resized to a fixed 64×64 to make similarity comparable.
    """

    H, W = tile.shape[:2]
    sf_h = int(H * strip_frac)
    sf_w = int(W * strip_frac)

    # convert to grayscale for NCC
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    strips = {
        "top": gray[0:sf_h, :],
        "bottom": gray[H - sf_h:H, :],
        "left": gray[:, 0:sf_w],
        "right": gray[:, W - sf_w:W]
    }

    FIX = 64  # normalize strip size

    for k in strips:
        strips[k] = cv2.resize(strips[k], (FIX, FIX), interpolation=cv2.INTER_AREA)

    return strips
