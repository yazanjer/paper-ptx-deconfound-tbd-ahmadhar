"""Tiny synthetic datasets in the prepared format, with a planted chest-tube shortcut:
tubes appear mostly in pneumothorax images, and the 'air' signal is a faint dark crescent."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ptx.data.prepare import COLS


def make(root: Path, n=(120, 60, 60, 30), size=64, seed=0):
    rng = np.random.default_rng(seed)
    specs = [("candid", n[0], True, True, True), ("siim", n[1], True, False, True),
             ("nih", n[2], False, False, True), ("ptx498", n[3], True, False, False)]
    for ds, N, has_mask, has_tube_mask, has_neg in specs:
        d = root / ds
        for k in ("img", "ptx", "tube"):
            (d / k).mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(N):
            ptx = int(rng.random() < 0.5) if has_neg else 1
            tube = int(rng.random() < (0.7 if ptx else 0.15))
            img = (rng.normal(128, 20, (size, size))).clip(0, 255).astype(np.uint8)
            pm = np.zeros((size, size), np.uint8)
            tm = np.zeros((size, size), np.uint8)
            if ptx:
                cv2.ellipse(pm, (size // 4, size // 4), (size // 8, size // 5), 0, 0, 360, 1, -1)
                img[pm > 0] = (img[pm > 0] * 0.8).astype(np.uint8)
            if tube:
                x = int(rng.integers(size // 2, size - 4))
                cv2.line(tm, (x, size - 1), (x - 10, size // 3), 1, 2)
                img[tm > 0] = 250
            iid = f"{ds}{i:04d}"
            cv2.imwrite(str(d / "img" / f"{iid}.png"), img)
            if has_mask:
                cv2.imwrite(str(d / "ptx" / f"{iid}.png"), pm * 255)
            if has_tube_mask:
                cv2.imwrite(str(d / "tube" / f"{iid}.png"), tm * 255)
            treated = tube if ds in ("candid", "siim", "nih") else -1
            rows.append({"image_id": iid, "dataset": ds, "patient_id": f"{ds}p{i // 2}", "ptx": ptx,
                         "tube": tube if ds != "nih" else -1, "device": tube if ds == "nih" else -1,
                         "treated": treated, "treated_source": "synthetic", "has_ptx_mask": int(has_mask),
                         "has_tube_mask": int(has_tube_mask), "spacing_mm": 0.7 if ds != "ptx498" else np.nan,
                         "site": ds, "view": "PA",
                         "source_split": ["train", "validation", "test"][i % 3] if ds == "nih" else ""})
        pd.DataFrame(rows, columns=COLS).to_csv(d / "manifest.csv", index=False)
