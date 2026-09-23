"""PyTorch dataset with optional chest-tube inpainting (component C1)."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_table(data: Path, dataset: str, splits_dir: Path, split) -> pd.DataFrame:
    man = pd.read_csv(data / dataset / "manifest.csv", dtype={"image_id": str})
    sp = pd.read_csv(splits_dir / f"{dataset}.csv", dtype={"image_id": str})
    df = man.merge(sp, on="image_id", how="inner")
    splits = [split] if isinstance(split, str) else list(split)
    df = df[df.split.isin(splits)].reset_index(drop=True)
    for c in ("tube", "device", "treated"):
        df[c] = df[c].fillna(-1).astype(int)
    return df


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "telea", dilate: int = 7) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return img
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    if method == "telea":
        return cv2.inpaint(img, m * 255, 5, cv2.INPAINT_TELEA)
    if method == "mean":  # simple local-mean fill (ablation)
        blur = cv2.blur(img, (31, 31))
        out = img.copy()
        out[m > 0] = blur[m > 0]
        return out
    raise ValueError(method)


class CXRDataset(Dataset):
    def __init__(self, table: pd.DataFrame, data: Path, size: int = 512, train: bool = False,
                 inpaint_p: float = 0.0, inpaint_method: str = "telea", occlusion: str | None = None):
        self.t = table.reset_index(drop=True)
        self.data = Path(data)
        self.size = size
        self.train = train
        self.inpaint_p = inpaint_p
        self.inpaint_method = inpaint_method
        self.occlusion = occlusion  # None | "tube_only" (audit: keep only the dilated tube region)
        self.tube_bank = []
        if inpaint_p > 0:
            self.tube_bank = [i for i, r in self.t.iterrows() if r.has_tube_mask == 1 and r.tube == 1]

    def __len__(self):
        return len(self.t)

    def _read(self, ds, kind, iid, flag=cv2.IMREAD_GRAYSCALE):
        p = self.data / ds / kind / f"{iid}.png"
        if not p.exists():
            return None
        a = cv2.imread(str(p), flag)
        if a.shape[0] != self.size:
            interp = cv2.INTER_AREA if kind == "img" else cv2.INTER_NEAREST
            a = cv2.resize(a, (self.size, self.size), interpolation=interp)
        return a

    def _aug(self, img, masks):
        s = self.size
        ang = random.uniform(-10, 10)
        sc = random.uniform(0.9, 1.1)
        tx, ty = random.uniform(-0.05, 0.05) * s, random.uniform(-0.05, 0.05) * s
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        masks = [cv2.warpAffine(m, M, (s, s), flags=cv2.INTER_NEAREST) if m is not None else None for m in masks]
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            masks = [np.ascontiguousarray(m[:, ::-1]) if m is not None else None for m in masks]
        a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img, masks

    def __getitem__(self, i):
        r = self.t.iloc[i]
        img = self._read(r.dataset, "img", r.image_id)
        ptx = self._read(r.dataset, "ptx", r.image_id) if r.has_ptx_mask == 1 else None
        tube = self._read(r.dataset, "tube", r.image_id) if r.has_tube_mask == 1 else None
        tube_label = int(r.treated)  # chest tube if known, else any visible device (NIH), -1 unknown
        inpainted = 0
        if self.train and self.inpaint_p > 0 and random.random() < self.inpaint_p:
            if tube is not None and tube_label == 1:
                img = inpaint(img, tube, self.inpaint_method)
                tube = np.zeros_like(tube)
                tube_label = 0
            elif self.tube_bank:  # decoy: inpaint a random tube-shaped region so inpainting is not a signal
                j = random.choice(self.tube_bank)
                rr = self.t.iloc[j]
                decoy = self._read(rr.dataset, "tube", rr.image_id)
                if ptx is not None:
                    decoy = decoy * (ptx == 0)
                img = inpaint(img, decoy, self.inpaint_method)
            inpainted = 1
        if self.occlusion == "tube_only" and tube is not None:
            keep = cv2.dilate((tube > 0).astype(np.uint8), np.ones((25, 25), np.uint8))
            fill = int(img.mean())
            img = np.where(keep > 0, img, fill).astype(np.uint8)
        if self.train:
            img, (ptx, tube) = self._aug(img, [ptx, tube])
        x = (np.repeat(img[..., None], 3, -1).astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        z = np.zeros((self.size, self.size), np.float32)
        ptx_m = torch.from_numpy(((ptx > 0).astype(np.float32) if ptx is not None else z)[None])
        tube_m = torch.from_numpy(((tube > 0).astype(np.float32) if tube is not None else z)[None])
        return {
            "x": x, "ptx_mask": ptx_m, "has_ptx_mask": torch.tensor(float(r.has_ptx_mask == 1)),
            "ptx": torch.tensor(float(r.ptx)), "tube_mask": tube_m,
            "has_tube_mask": torch.tensor(float(tube is not None)),
            "tube": torch.tensor(float(tube_label)),
            "group": torch.tensor(int(r.ptx) * 2 + (1 if int(r.treated) == 1 else 0)),
            "inpainted": torch.tensor(inpainted), "idx": torch.tensor(i),
        }
