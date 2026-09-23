"""Train one model (one config x one seed), pick the operating threshold on dev validation,
then write predictions for every evaluation set listed in the config.

    python -m ptx.train --config configs/runs/full.yaml --seed 0 --out results/runs

Outputs in <out>/<run_name>/seed<k>/:
    config.yaml, train_log.csv, threshold.json, preds_<dataset>_<split>.csv,
    [feats_<dataset>_<split>.npz], [preds_<dataset>_<split>_tubeonly.csv], DONE
    best.pt (weights; excluded from git, uploaded as a release asset)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ptx.data.dataset import CXRDataset, load_table
from ptx.eval.metrics import auroc, dice, sens_at_spec, threshold_at_specificity
from ptx.eval.size import size_features
from ptx.models.net import build_model


# --------------------------------------------------------------------------- config
def deep_update(a, b):
    for k, v in b.items():
        a[k] = deep_update(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def load_config(path):
    cfg = yaml.safe_load(open(path))
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(Path(path).parent / base), cfg)
    return cfg


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


# --------------------------------------------------------------------------- losses
def soft_dice_loss(logits, target, weight):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).flatten(1).sum(1) + 1
    den = (p + target).flatten(1).sum(1) + 1
    l = 1 - num / den
    return (l * weight).sum() / weight.sum().clamp(min=1)


def seg_loss(logits, target, has):
    if has.sum() == 0:
        return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
    bce = (bce * has).sum() / has.sum()
    pos = has * (target.flatten(1).sum(1) > 0).float()
    return bce + soft_dice_loss(logits, target, pos) if pos.sum() > 0 else bce


class GroupDRO:
    def __init__(self, n_groups=4, eta=0.01, device="cpu"):
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.eta = eta

    def __call__(self, per_sample, groups):
        losses = torch.zeros_like(self.q)
        for g in range(len(self.q)):
            m = groups == g
            if m.any():
                losses[g] = per_sample[m].mean()
        self.q = self.q * torch.exp(self.eta * losses.detach())
        self.q = self.q / self.q.sum()
        return (self.q * losses).sum()


def compute_loss(out, b, cfg, dro=None):
    L = cfg["loss"]
    terms = {}
    per = F.binary_cross_entropy_with_logits(out["cls"], b["ptx"], reduction="none")
    terms["cls"] = dro(per, b["group"]) if dro is not None else per.mean()
    if "seg_ptx" in out and L.get("w_seg", 1.0) > 0:
        terms["seg"] = L.get("w_seg", 1.0) * seg_loss(out["seg_ptx"], b["ptx_mask"], b["has_ptx_mask"])
    known = b["tube"] >= 0
    if "seg_tube" in out:
        terms["tube_seg"] = L.get("w_tube_seg", 0.5) * seg_loss(out["seg_tube"], b["tube_mask"], b["has_tube_mask"])
    if "adv" in out and known.any():
        terms["adv"] = L.get("w_adv", 1.0) * F.binary_cross_entropy_with_logits(out["adv"][known], b["tube"][known])
    if "aux_cls" in out and known.any():
        terms["aux_cls"] = L.get("w_aux_cls", 0.5) * F.binary_cross_entropy_with_logits(out["aux_cls"][known], b["tube"][known])
    return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, ds, device, bs=16, workers=4, want_feats=False, want_seg=True):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    rows, feats = [], []
    t = ds.t
    for b in dl:
        x = b["x"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        logit = out["cls"].float().cpu().numpy()
        seg = torch.sigmoid(out["seg_ptx"].float()).cpu().numpy()[:, 0] if ("seg_ptx" in out and want_seg) else None
        if want_feats:
            feats.append(out["feat"].float().cpu().numpy())
        for j, i in enumerate(b["idx"].numpy()):
            r = t.iloc[i]
            row = {"image_id": r.image_id, "dataset": r.dataset, "patient_id": r.patient_id, "ptx": int(r.ptx),
                   "treated": int(r.treated), "treated_source": r.treated_source, "logit": float(logit[j]),
                   "score": float(1 / (1 + math.exp(-logit[j])))}
            if seg is not None:
                pm = seg[j] > 0.5
                sp = r.spacing_mm
                for k, v in size_features(pm, sp).items():
                    row[f"pred_{k}"] = v
                if r.has_ptx_mask == 1:
                    gm = b["ptx_mask"][j, 0].numpy() > 0.5
                    for k, v in size_features(gm, sp).items():
                        row[f"gt_{k}"] = v
                    row["dice"] = dice(pm, gm)
            rows.append(row)
    df = pd.DataFrame(rows)
    return (df, np.concatenate(feats)) if want_feats else (df, None)


def val_metrics(df):
    y, s = df.ptx.values, df.score.values
    m = {"auroc": auroc(y, s), "sens95": sens_at_spec(y, s, 0.95)}
    unt = df[(df.treated == 0) | (df.ptx == 0)]
    if (df.treated >= 0).any() and (unt.ptx == 1).any():
        m["sens95_untreated"] = sens_at_spec(unt.ptx.values, unt.score.values, 0.95)
    if "dice" in df and df.ptx.sum() > 0:
        m["dice_pos"] = float(df.loc[df.ptx == 1, "dice"].mean())
    return m


# --------------------------------------------------------------------------- main
def run(cfg, seed, out_root: Path, max_epochs=None, limit=None):
    name = cfg["name"]
    od = out_root / name / f"seed{seed}"
    if (od / "DONE").exists():
        print(f"skip {od} (DONE)")
        return od
    od.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(od / "config.yaml", "w"))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, data, splits = cfg["data"], Path(cfg["data"]["root"]), Path(cfg["data"]["splits"])
    tr = load_table(data, D["dev"], splits, D.get("train_split", "train"))
    va = load_table(data, D["dev"], splits, D.get("val_split", "val"))
    if limit:
        tr, va = tr.sample(min(limit, len(tr)), random_state=seed), va.sample(min(limit, len(va)), random_state=seed)
    T = cfg["train"]
    size = D.get("size", 512)
    ds_tr = CXRDataset(tr, data, size, train=True, inpaint_p=T.get("inpaint_p", 0.0), inpaint_method=T.get("inpaint_method", "telea"))
    ds_va = CXRDataset(va, data, size)
    sampler = None
    if T.get("balance_pos", 0) > 0:  # oversample positives to the requested fraction
        w = np.where(tr.ptx.values == 1, T["balance_pos"] / max(tr.ptx.mean(), 1e-6), (1 - T["balance_pos"]) / max(1 - tr.ptx.mean(), 1e-6))
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
    workers = T.get("workers", 8)
    dl = DataLoader(ds_tr, batch_size=T["batch_size"], shuffle=sampler is None, sampler=sampler,
                    num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=T["lr"], weight_decay=T.get("weight_decay", 1e-4))
    epochs = max_epochs or T["epochs"]
    steps = epochs * len(dl)
    warm = min(len(dl), steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / max(warm, 1) if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dro = GroupDRO(device=device) if T.get("group_dro", False) else None
    sel = T.get("select_metric", "sens95_untreated")
    best, log = -1e9, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        agg = {}
        for b in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(b["x"])
            out = {k: v.float() for k, v in out.items()}
            loss, terms = compute_loss(out, b, cfg, dro)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in terms.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        vdf, _ = predict(model, ds_va, device, workers=workers)
        vm = val_metrics(vdf)
        score = vm.get(sel, vm["auroc"])
        score = -1 if score != score else score
        log.append({"epoch": ep, **agg, **{f"val_{k}": v for k, v in vm.items()}, "minutes": (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(od / "train_log.csv", index=False)
        print(f"[{name} s{seed}] ep{ep} " + " ".join(f"{k}={v:.4f}" for k, v in log[-1].items() if isinstance(v, float)))
        if score > best:
            best = score
            torch.save(model.state_dict(), od / "best.pt")
    model.load_state_dict(torch.load(od / "best.pt", map_location=device))
    vdf, _ = predict(model, ds_va, device, workers=workers)
    vdf.to_csv(od / f"preds_{D['dev']}_val.csv", index=False)
    thr = threshold_at_specificity(vdf.ptx.values, vdf.score.values, 0.95)
    json.dump({"threshold_spec95": thr, "val": val_metrics(vdf), "select_metric": sel, "best": best},
              open(od / "threshold.json", "w"), indent=2)
    evaluate_sets(model, cfg, od, device, workers, limit)
    (od / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return od


def evaluate_sets(model, cfg, od, device, workers, limit=None):
    data, splits = Path(cfg["data"]["root"]), Path(cfg["data"]["splits"])
    unlocked = os.environ.get("PTX_UNLOCK_EXTERNAL") == "1"
    for spec in cfg.get("eval", {}).get("sets", []):
        ds_name, split = spec["dataset"], spec["split"]
        if spec.get("external", False) and not unlocked:
            print(f"external set {ds_name}/{split} locked (set PTX_UNLOCK_EXTERNAL=1 at Gate 2)")
            continue
        if not (splits / f"{ds_name}.csv").exists():
            print(f"skip {ds_name}: no split file")
            continue
        t = load_table(data, ds_name, splits, split)
        if limit:
            t = t.head(limit)
        size = cfg["data"].get("size", 512)
        df, feats = predict(model, CXRDataset(t, data, size), device, workers=workers, want_feats=cfg.get("audit", False))
        df.to_csv(od / f"preds_{ds_name}_{split}.csv", index=False)
        if feats is not None:
            np.savez_compressed(od / f"feats_{ds_name}_{split}.npz", feats=feats.astype(np.float16), image_id=df.image_id.values)
        if cfg.get("audit", False) and (t.has_tube_mask == 1).any():
            tt = t[(t.has_tube_mask == 1) & (t.tube == 1)]
            occ, _ = predict(model, CXRDataset(tt, data, size, occlusion="tube_only"), device, workers=workers, want_seg=False)
            occ.to_csv(od / f"preds_{ds_name}_{split}_tubeonly.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/runs"))
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--limit", type=int, help="debug: subsample N images per split")
    a = ap.parse_args()
    run(load_config(a.config), a.seed, a.out, a.max_epochs, a.limit)


if __name__ == "__main__":
    main()
