"""Render 24 random positive overlays per dataset (PTX red, tube green) to verify RLE decoding
and orientation by eye before any training."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path("results/mask_checks"))
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for ds in ["candid", "siim", "ptx498"]:
    f = a.data / ds / "manifest.csv"
    if not f.exists():
        continue
    m = pd.read_csv(f, dtype={"image_id": str})
    m = m[m.ptx == 1].sample(min(24, int(m.ptx.sum())), random_state=0)
    tiles = []
    for iid in m.image_id:
        img = cv2.imread(str(a.data / ds / "img" / f"{iid}.png"), 0)
        rgb = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_GRAY2BGR)
        for kind, col in (("ptx", (0, 0, 255)), ("tube", (0, 255, 0))):
            p = a.data / ds / kind / f"{iid}.png"
            if p.exists():
                mk = cv2.resize(cv2.imread(str(p), 0), (256, 256)) > 0
                cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, col, 1)
        tiles.append(rgb)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(str(a.out / f"{ds}_overlays.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", a.out / f"{ds}_overlays.jpg")
