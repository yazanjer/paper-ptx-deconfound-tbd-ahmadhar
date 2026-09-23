"""Plan B: image-level 'visible device' pseudo-labels for SIIM-ACR.

Trains a device classifier on the NIH cohort's released TRAIN split only (visible-device
labels, uncertain excluded), picks the cut-off on its VALIDATION split, predicts every SIIM-ACR
image and writes `device`, `treated`, `treated_source`, `device_prob` into the SIIM manifest.
The NIH TEST split is never touched. A review sheet of SIIM positives is written for manual checks.

    python -m ptx.data.device_labels --data /workspace/data --splits splits --target siim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ptx.data.dataset import CXRDataset, load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--source", default="nih")
    ap.add_argument("--target", default="siim")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/device_labels"))
    a = ap.parse_args()
    import timm
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_table(a.data, a.source, a.splits, "pilot_train")
    va = load_table(a.data, a.source, a.splits, "pilot_val")
    tr, va = tr[tr.device >= 0].reset_index(drop=True), va[va.device >= 0].reset_index(drop=True)
    model = timm.create_model(a.backbone, pretrained=bool(a.pretrained), num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    dl = DataLoader(CXRDataset(tr, a.data, a.size, train=True), batch_size=32, shuffle=True,
                    num_workers=a.workers, drop_last=len(tr) > 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=max(1, a.epochs * len(dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    @torch.no_grad()
    def probs(t):
        model.eval()
        out = []
        for b in DataLoader(CXRDataset(t, a.data, a.size), batch_size=64, num_workers=a.workers):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(b["x"].to(dev))
            out.append(torch.sigmoid(lg.float()).cpu().numpy()[:, 0])
        return np.concatenate(out) if out else np.array([])

    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            x, y = b["x"].to(dev), b["tube"].to(dev)  # 'tube' carries the treated/device label
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(x)
            loss = F.binary_cross_entropy_with_logits(lg.float()[:, 0], y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = probs(va)
        auc = roc_auc_score(va.device.values, pv) if va.device.nunique() > 1 else float("nan")
        print(f"device ep{ep} loss {loss.item():.4f} val AUC {auc:.4f}")
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv = probs(va)
    y = va.device.values
    cands = np.unique(np.quantile(pv, np.linspace(0.02, 0.98, 49)))
    bal = [0.5 * ((pv[y == 1] >= c).mean() + (pv[y == 0] < c).mean()) for c in cands]
    thr = float(cands[int(np.argmax(bal))])
    stats = {"val_auc": float(roc_auc_score(y, pv)), "threshold": thr, "val_balanced_acc": float(max(bal)),
             "val_sens": float((pv[y == 1] >= thr).mean()), "val_spec": float((pv[y == 0] < thr).mean()),
             "n_train": int(len(tr)), "n_val": int(len(va))}
    print(stats)

    man_p = a.data / a.target / "manifest.csv"
    man = pd.read_csv(man_p, dtype={"image_id": str})
    for c in ("tube", "device", "treated"):
        man[c] = man[c].fillna(-1).astype(int)
    man["has_ptx_mask"] = man.has_ptx_mask.fillna(0).astype(int)
    man["has_tube_mask"] = man.has_tube_mask.fillna(0).astype(int)
    p = probs(man.assign(split="all"))
    man["device_prob"] = p
    man["device"] = (p >= thr).astype(int)
    man["treated"] = man["device"]
    man["treated_source"] = "pseudo_device_cls"
    man.to_csv(man_p, index=False)
    stats.update({"target": a.target, "target_n": int(len(man)), "target_device_rate": float(man.device.mean()),
                  "target_device_rate_ptx_pos": float(man.loc[man.ptx == 1, "device"].mean()),
                  "target_device_rate_ptx_neg": float(man.loc[man.ptx == 0, "device"].mean())})
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(a.out / "device_classifier.json", "w"), indent=2)
    man[man.ptx == 1][["image_id", "device", "device_prob"]].assign(reviewer_device="", reviewer="").to_csv(
        a.out / f"{a.target}_positives_review.csv", index=False)
    torch.save(model.state_dict(), a.out / f"device_{a.backbone}.pt")
    print(f"{a.target}: device rate {stats['target_device_rate']:.3f} "
          f"(PTX+ {stats['target_device_rate_ptx_pos']:.3f}, PTX- {stats['target_device_rate_ptx_neg']:.3f})")


if __name__ == "__main__":
    main()
