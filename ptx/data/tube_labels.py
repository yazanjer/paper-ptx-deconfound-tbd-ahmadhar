"""Chest-tube pseudo-labels for datasets without tube annotations (SIIM-ACR).

Trains a tube segmenter on the DEV training split only (CANDID-PTX tube masks), predicts on the
target dataset, and writes `tube`, `treated`, `treated_source` back into its manifest. It also
writes a review sheet of every target PTX-positive so a human can confirm each label
(pre-registered: all external positives are checked manually).

    python -m ptx.data.tube_labels --data /workspace/data --splits splits --dev candid --target siim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ptx.data.dataset import CXRDataset, load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--dev", default="candid")
    ap.add_argument("--target", default="siim")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--area-thr", type=float, default=2e-4, help="min predicted tube area fraction for tube=1")
    ap.add_argument("--out", type=Path, default=Path("results/tube_labels"))
    ap.add_argument("--encoder-weights", default="imagenet")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_table(a.data, a.dev, a.splits, "train")
    va = load_table(a.data, a.dev, a.splits, "val")
    tr, va = tr[tr.has_tube_mask == 1], va[va.has_tube_mask == 1]
    model = smp.Unet("resnet34", encoder_weights=None if a.encoder_weights == "none" else a.encoder_weights, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dl = DataLoader(CXRDataset(tr, a.data, a.size, train=True), batch_size=16, shuffle=True, num_workers=a.workers, drop_last=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            x, y = b["x"].to(dev), b["tube_mask"].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(x)
            lg = lg.float()
            p = torch.sigmoid(lg)
            loss = F.binary_cross_entropy_with_logits(lg, y) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        print(f"tube ep{ep} loss {loss.item():.4f}")

    @torch.no_grad()
    def areas(t):
        model.eval()
        out = []
        for b in DataLoader(CXRDataset(t, a.data, a.size), batch_size=16, num_workers=a.workers):
            p = torch.sigmoid(model(b["x"].to(dev)).float()).cpu().numpy()[:, 0]
            out += list((p > 0.5).mean((1, 2)))
        return np.array(out)

    # calibrate on dev val: report image-level accuracy of the area rule
    va_area = areas(va)
    if va.tube.nunique() > 1:  # pick the area cut-off with best balanced accuracy on dev validation
        cands = np.unique(np.r_[a.area_thr, np.quantile(va_area, np.linspace(0.05, 0.95, 37))])
        bal = [0.5 * ((va_area[va.tube.values == 1] >= c).mean() + (va_area[va.tube.values == 0] < c).mean()) for c in cands]
        a.area_thr = float(cands[int(np.argmax(bal))])
    acc = ((va_area >= a.area_thr).astype(int) == va.tube.values).mean()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(va.tube.values, va_area) if va.tube.nunique() > 1 else float("nan")
    print(f"dev-val tube detection: AUC {auc:.3f}, accuracy at area>={a.area_thr}: {acc:.3f}")

    man_p = a.data / a.target / "manifest.csv"
    man = pd.read_csv(man_p, dtype={"image_id": str})
    man["has_ptx_mask"] = man.has_ptx_mask.fillna(0).astype(int)
    man["has_tube_mask"] = 0
    man["tube"], man["device"], man["treated"] = man.tube.fillna(-1), man.device.fillna(-1), man.treated.fillna(-1)
    ar = areas(man.assign(split="all"))
    man["tube_area_pred"] = ar
    man["tube"] = (ar >= a.area_thr).astype(int)
    man["treated"] = man["tube"]
    man["treated_source"] = "pseudo_tube_seg"
    man.to_csv(man_p, index=False)
    a.out.mkdir(parents=True, exist_ok=True)
    review = man[man.ptx == 1][["image_id", "tube", "tube_area_pred"]].assign(reviewer_tube="", reviewer="")
    review.to_csv(a.out / f"{a.target}_positives_review.csv", index=False)
    (a.out / "tube_detector_devval.txt").write_text(f"AUC {auc:.4f}\naccuracy {acc:.4f}\narea_thr {a.area_thr}\n")
    torch.save(model.state_dict(), a.out / "tube_seg_r34.pt")
    print(f"{a.target}: {int(man.tube.sum())} / {len(man)} images pseudo-labelled with tube; review sheet written")


if __name__ == "__main__":
    main()
