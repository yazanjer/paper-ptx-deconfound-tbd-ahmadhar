"""Render example radiographs and model heatmaps for the manuscript (no training, no metrics).

Examples are chosen by fixed, pre-stated rules (not by visual inspection):
  NIH test (primary external), using the proposed model's seed-averaged score:
    untreated_tp   untreated positive with the highest score
    untreated_fn   untreated positive with the lowest score
    device_pos     device-present positive with the median score
    device_neg_fp  device-present NEGATIVE with the highest score (shortcut candidate)
    clean_neg      device-free negative with the median score
  SIIM-ACR internal test: 2 positives with masks, random (seed 0)
  PTX-498: 1 positive per site, random (seed 0)

For each example and model we save, averaged over that model's seeds:
  seg_<run>.png   pneumothorax probability map (sigmoid of the segmentation logits)
  cam_<run>.png   Grad-CAM of the image-level logit on the last encoder stage
Outputs: results/paper_figs/<example>/{img.png, mask.png?, seg_*.png, cam_*.png}, index.json

    python scripts/render_paper_figs.py --data /workspace/data --runs results/runs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptx.data.dataset import CXRDataset, load_table  # noqa: E402
from ptx.models.net import build_model  # noqa: E402

MODELS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro"]


def seed_avg_scores(runs: Path, run: str, ds: str):
    fs = sorted(glob.glob(str(runs / run / "seed*" / f"preds_{ds}_test.csv")))
    dfs = [pd.read_csv(f, dtype={"image_id": str}).set_index("image_id") for f in fs]
    base = dfs[0].copy()
    base["score_avg"] = np.mean([d.loc[base.index, "score"].values for d in dfs], axis=0)
    return base.reset_index()


def pick_examples(a):
    nih = seed_avg_scores(a.runs, "pb_full", "nih")
    nih = nih[nih.treated >= 0]
    up = nih[(nih.ptx == 1) & (nih.treated == 0)].sort_values("score_avg")
    dp = nih[(nih.ptx == 1) & (nih.treated == 1)].sort_values("score_avg")
    dn = nih[(nih.ptx == 0) & (nih.treated == 1)].sort_values("score_avg")
    cn = nih[(nih.ptx == 0) & (nih.treated == 0)].sort_values("score_avg")
    ex = [("nih", "untreated_tp", up.iloc[-1].image_id), ("nih", "untreated_fn", up.iloc[0].image_id),
          ("nih", "device_pos", dp.iloc[len(dp) // 2].image_id), ("nih", "device_neg_fp", dn.iloc[-1].image_id),
          ("nih", "clean_neg", cn.iloc[len(cn) // 2].image_id)]
    rng = np.random.default_rng(0)
    siim = load_table(a.data, "siim", a.splits, "test_internal")
    siim = siim[(siim.ptx == 1) & (siim.has_ptx_mask == 1)]
    for i, iid in enumerate(rng.choice(siim.image_id.values, 2, replace=False)):
        ex.append(("siim", f"siim_pos{i + 1}", iid))
    ptx = load_table(a.data, "ptx498", a.splits, "test")
    for site in sorted(ptx.site.dropna().unique()):
        ex.append(("ptx498", f"ptx498_{site}", rng.choice(ptx[ptx.site == site].image_id.values)))
    return ex


def gradcam(model, x):
    store = {}

    def hook(_m, _i, out):
        store["a"] = out[-1]
        out[-1].retain_grad()

    h = model.seg.encoder.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    out = model(x)
    out["cls"].sum().backward()
    h.remove()
    a, g = store["a"], store["a"].grad
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    seg = torch.sigmoid(out["seg_ptx"].detach().float())[0, 0]
    return seg.cpu().numpy(), cam.detach().cpu().numpy(), float(torch.sigmoid(out["cls"].detach())[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/paper_figs"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = pick_examples(a)
    tables = {ds: pd.concat([load_table(a.data, ds, a.splits, s) for s in (["test"] if ds != "siim" else ["test_internal"])])
              for ds in {e[0] for e in ex}}
    items, index = [], []
    for ds, name, iid in ex:
        t = tables[ds]
        row = t[t.image_id == iid].iloc[[0]].assign(split="all")
        item = CXRDataset(row, a.data, 512)[0]
        od = a.out / name
        od.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(od / "img.png"), cv2.imread(str(a.data / ds / "img" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE))
        if item["has_ptx_mask"] > 0:
            cv2.imwrite(str(od / "mask.png"), (item["ptx_mask"][0].numpy() * 255).astype(np.uint8))
        items.append((od, item["x"][None].to(dev)))
        index.append({"name": name, "dataset": ds, "image_id": str(iid), "ptx": int(row.ptx.iloc[0]),
                      "treated": int(row.treated.iloc[0]), "scores": {}})
    for run in MODELS:
        acc = [([], [], []) for _ in items]
        for ck in sorted(glob.glob(str(a.runs / run / "seed*" / "best.pt"))):
            cfg = yaml.safe_load(open(Path(ck).parent / "config.yaml"))
            m = build_model(cfg).to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev))
            m.eval()
            for (od, x), (segs, cams, scores) in zip(items, acc):
                s, c, p = gradcam(m, x)
                segs.append(s), cams.append(c), scores.append(p)
            del m
        for (od, _), (segs, cams, scores), rec in zip(items, acc, index):
            cv2.imwrite(str(od / f"seg_{run}.png"), (np.mean(segs, 0) * 255).astype(np.uint8))
            cv2.imwrite(str(od / f"cam_{run}.png"), (np.mean(cams, 0) * 255).astype(np.uint8))
            rec["scores"][run] = float(np.mean(scores))
        print("rendered", run, len(acc[0][2]), "seeds", flush=True)
    json.dump(index, open(a.out / "index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
