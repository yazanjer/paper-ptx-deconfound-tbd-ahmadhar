"""Run-length encoding helpers for SIIM-ACR and CANDID-PTX masks.

SIIM-ACR (2019) uses a *relative* column-major RLE: each start is an offset from
the end of the previous run.  CANDID-PTX uses the same Kaggle-style layout; the
`mode` argument lets you switch to absolute starts if a sanity check fails
(see scripts/check_masks.py, which renders overlays for visual verification).
"""
from __future__ import annotations

import numpy as np


def rle_decode(rle: str, height: int = 1024, width: int = 1024, mode: str = "relative") -> np.ndarray:
    """Decode one RLE string to a uint8 {0,1} mask of shape (height, width)."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None:
        return mask.reshape(width, height).T
    rle = str(rle).strip()
    if rle in ("", "-1", "nan", "None"):
        return mask.reshape(width, height).T
    arr = np.asarray([int(x) for x in rle.split()], dtype=np.int64)
    starts, lengths = arr[0::2], arr[1::2]
    pos = 0
    for s, l in zip(starts, lengths):
        if mode == "relative":
            pos += s
            mask[pos:pos + l] = 1
            pos += l
        elif mode == "absolute":
            mask[s - 1:s - 1 + l] = 1  # 1-indexed absolute starts
        else:
            raise ValueError(f"unknown RLE mode {mode}")
    # column-major layout -> transpose
    return mask.reshape(width, height).T


def rle_encode(mask: np.ndarray, mode: str = "relative") -> str:
    """Inverse of rle_decode (used in tests and for nnU-Net export round-trips)."""
    flat = mask.T.flatten().astype(np.uint8)
    if flat.sum() == 0:
        return "-1"
    padded = np.concatenate([[0], flat, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0]
    starts, ends = runs[0::2], runs[1::2]
    lengths = ends - starts
    out = []
    prev_end = 0
    for s, l in zip(starts, lengths):
        if mode == "relative":
            out += [int(s - prev_end), int(l)]
            prev_end = s + l
        else:
            out += [int(s + 1), int(l)]
    return " ".join(map(str, out))


def union_rles(rles, height=1024, width=1024, mode="relative") -> np.ndarray:
    m = np.zeros((height, width), dtype=np.uint8)
    for r in rles:
        m |= rle_decode(r, height, width, mode)
    return m
