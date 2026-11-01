"""Revision R1 (IEEE Access resubmission, Access-2026-49559): GPU analyses on the released checkpoints.

No training. Every model is re-loaded from its `best.pt` and run once more on the same images.
Outputs (results/r1/):
  repro.csv                     max |logit - stored logit| per run/seed/dataset (reproducibility from weights)
  topk/<run>__<seed>__<ds>.csv.gz  per image: logit, top-k overlap with the reference mask, top-k centroid
  topk_maps.npz                 32x32 histograms of top-k locations, summed per run/dataset/group
  probe.json                    linear device probes on pooled encoder features (image-level and patient-grouped CV)
  complexity.csv                parameters, FLOPs, inference latency/throughput, peak memory, train-step cost
  nih_image_stats.csv, siim_image_stats.csv  intensity statistics per image (threshold-transfer analysis)

    python scripts/r1_gpu.py --data /workspace/data --runs results/runs --out results/r1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ptx.data.dataset import CXRDataset, load_table
from ptx.models.net import build_model

SEG_RUNS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_sens_lambda001", "pb_sens_lambda1",
            "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls"]
PROBE_RUNS = ["pb_bl_raddino_probe", "pb_bl_raddino_ft"]
SETS = [("siim", "test_internal"), ("nih", "test"), ("ptx498", "test")]
DIL = 25  # px at 512 (about 5% of the image width) tolerance band around the reference mask
GRID = 32


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def table(data: Path, splits: Path, ds: str, split: str, siim_labels: pd.DataFrame | None):
    t = load_table(data, ds, splits, split)
    if ds == "siim" and siim_labels is not None:  # Plan-B pseudo device labels (as in training)
        t = t.drop(columns=[c for c in ("device", "treated", "treated_source") if c in t]).merge(
            siim_labels[["image_id", "device"]], on="image_id", how="left")
        t["device"] = t.device.fillna(-1).astype(int)
        t["treated"], t["treated_source"] = t.device, "pseudo_device_cls"
    return t


@torch.no_grad()
def run_model(model, ds, dev, want_seg: bool, bs=16, workers=6):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    k_frac = getattr(model, "topk_frac", 0.005)
    rows, feats = [], []
    for b in dl:
        x = b["x"].to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x)
        logit = out["cls"].float().cpu().numpy()
        feats.append(out["feat"].float().cpu().numpy())
        if want_seg:
            z = out["seg_ptx"].float()[:, 0]  # pixel logits Z
            n, h, w = z.shape
            k = max(1, int(k_frac * h * w))
            idx = z.flatten(1).topk(k, dim=1).indices.cpu().numpy()
            zmax = z.flatten(1).argmax(1).cpu().numpy()
            gt = b["ptx_mask"][:, 0].numpy() > 0.5
            has = b["has_ptx_mask"].numpy() > 0.5
        for j, i in enumerate(b["idx"].numpy()):
            r = ds.t.iloc[i]
            row = {"image_id": r.image_id, "patient_id": r.patient_id, "ptx": int(r.ptx), "treated": int(r.treated),
                   "logit": float(logit[j])}
            if want_seg:
                yy, xx = np.unravel_index(idx[j], (h, w))
                ay, ax = np.unravel_index(zmax[j], (h, w))
                row.update(topk_cy=float(yy.mean() / h), topk_cx=float(xx.mean() / w),
                           topk_spread=float(np.sqrt(yy.var() + xx.var()) / h), argmax_y=ay / h, argmax_x=ax / w)
                hist = np.histogram2d(yy, xx, bins=GRID, range=[[0, h], [0, w]])[0].astype(np.float32) / len(yy)
                row["_hist"] = hist
                if has[j] and gt[j].any():
                    g = gt[j]
                    gd = cv2.dilate(g.astype(np.uint8), np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)) > 0
                    row.update(gt_area_frac=float(g.mean()), topk_in_gt=float(g[yy, xx].mean()),
                               topk_in_gt_dil=float(gd[yy, xx].mean()), argmax_in_gt=int(g[ay, ax]),
                               argmax_in_gt_dil=int(gd[ay, ax]),
                               topk_any_in_gt=int(g[yy, xx].any()),
                               chance_in_gt=float(g.mean()), chance_in_gt_dil=float(gd.mean()))
            rows.append(row)
    return rows, np.concatenate(feats)


def probe_auc(X, y, groups=None, seed=0):
    k = y >= 0
    X, y = X[k], y[k]
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10:
        return None
    clf = LogisticRegression(max_iter=3000, C=0.1)
    out = {}
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    out["image_cv"] = auroc(y, p)  # identical to the pre-registered probe (held-out folds only)
    p = cross_val_predict(make_pipeline(StandardScaler(), clf), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    out["image_cv_std"] = auroc(y, p)
    if groups is not None:
        g = np.asarray(groups)[k]
        p = cross_val_predict(clf, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed), groups=g,
                              method="predict_proba")[:, 1]
        out["patient_cv"] = auroc(y, p)
    out["n"], out["n_pos"] = int(len(y)), int(y.sum())
    return out


def complexity(model, cfg, dev):
    from fvcore.nn import FlopCountAnalysis
    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "trainable_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}
    x1 = torch.randn(1, 3, 512, 512, device=dev)
    model.eval()
    try:
        fl = FlopCountAnalysis(model, x1)
        fl.unsupported_ops_warnings(False); fl.uncalled_modules_warnings(False)
        res["GFLOPs_fwd"] = fl.total() / 1e9  # fvcore counts one fused multiply-add as one flop (MACs)
    except Exception as e:  # noqa: BLE001
        res["GFLOPs_fwd"] = float("nan"); res["flops_error"] = str(e)[:120]
    for bs in (1, 16):
        x = torch.randn(bs, 3, 512, 512, device=dev)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(5):
                model(x)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20):
                model(x)
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 20
        res[f"latency_ms_bs{bs}"] = 1000 * dt
        res[f"img_per_s_bs{bs}"] = bs / dt
        res[f"peak_mem_GB_infer_bs{bs}"] = torch.cuda.max_memory_allocated() / 1e9
    bs = int(cfg["train"].get("batch_size", 8))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(bs, 3, 512, 512, device=dev)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(8):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = out["cls"].float().mean() + sum(v.float().mean() for k2, v in out.items() if k2 in ("seg_ptx", "adv", "aux_cls"))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    res["train_batch"] = bs
    res["train_step_ms"] = 1000 * float(np.median(times[3:]))
    res["peak_mem_GB_train"] = torch.cuda.max_memory_allocated() / 1e9
    model.eval()
    return res


def image_stats(data: Path, t: pd.DataFrame, ds: str):
    rows = []
    for r in t.itertuples():
        a = cv2.imread(str(data / ds / "img" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        lap = cv2.Laplacian(a, cv2.CV_32F)
        rows.append({"image_id": r.image_id, "mean": a.mean(), "std": a.std(), "p05": np.percentile(a, 5),
                     "p95": np.percentile(a, 95), "sharpness": lap.var(),
                     "frac_bright": float((a > 240).mean()), "frac_dark": float((a < 15).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--out", type=Path, default=Path("results/r1"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: images per set")
    ap.add_argument("--only", default="", help="comma list of runs")
    ap.add_argument("--drain", type=Path, default=Path("results/r1_inputs/siim_drain_labels.csv"))
    ap.add_argument("--nih-drain", type=Path, default=Path("results/r1_inputs/nih_neatx_drain.csv"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "topk").mkdir(exist_ok=True)
    lab_p = Path("results/device_labels/siim_device_labels.csv")
    siim_labels = pd.read_csv(lab_p, dtype={"image_id": str}) if lab_p.exists() else None
    tabs = {}
    for ds, sp in SETS:
        if (a.data / ds / "manifest.csv").exists() and (a.splits / f"{ds}.csv").exists():
            t = table(a.data, a.splits, ds, sp, siim_labels)
            if ds == "nih":  # only images that were actually prepared
                t = t[[(a.data / "nih" / "img" / f"{i}.png").exists() for i in t.image_id]].reset_index(drop=True)
            tabs[ds] = t.head(a.limit) if a.limit else t
            print(ds, len(tabs[ds]), flush=True)
    for ds in ("siim", "nih"):
        if ds in tabs and not (a.out / f"{ds}_image_stats.csv").exists():
            image_stats(a.data, tabs[ds], ds).to_csv(a.out / f"{ds}_image_stats.csv", index=False)
    drain = {}
    if a.drain.exists():
        d = pd.read_csv(a.drain, dtype={"image_id": str})
        drain["siim"] = dict(zip(d.image_id, d.Drain.astype(int)))
    if a.nih_drain.exists():
        d = pd.read_csv(a.nih_drain, dtype={"image_id": str})
        drain["nih"] = dict(zip(d.image_id, d.Drain.astype(int)))

    runs = [r for r in (a.only.split(",") if a.only else SEG_RUNS + PROBE_RUNS) if r]
    repro, probes, comp, maps = [], {}, [], {}
    if (a.out / "probe.json").exists():
        probes = json.load(open(a.out / "probe.json"))
    for run in runs:
        for sd in sorted(glob.glob(str(a.runs / run / "seed*"))):
            sd = Path(sd)
            if not (sd / "best.pt").exists():
                print("missing weights", sd, flush=True)
                continue
            tag = f"{run}__{sd.name}"
            if (a.out / f"DONE_{tag}").exists():
                continue
            cfg = yaml.safe_load(open(sd / "config.yaml"))
            model = build_model(cfg).to(dev)
            model.load_state_dict(torch.load(sd / "best.pt", map_location=dev))
            model.eval()
            seg = run in SEG_RUNS
            for ds, t in tabs.items():
                rows, feats = run_model(model, CXRDataset(t, a.data, 512), dev, want_seg=seg)
                hist = [r.pop("_hist", None) for r in rows]
                df = pd.DataFrame(rows)
                st = sd / f"preds_{ds}_{dict(SETS)[ds]}.csv"
                if st.exists():
                    old = pd.read_csv(st, dtype={"image_id": str})[["image_id", "logit"]].rename(columns={"logit": "logit_stored"})
                    m = df.merge(old, on="image_id")
                    repro.append({"run": run, "seed": sd.name, "dataset": ds, "n": len(m),
                                  "max_abs_logit_diff": float((m.logit - m.logit_stored).abs().max()),
                                  "auroc_now": auroc(m.ptx, m.logit) if ds != "ptx498" else float("nan"),
                                  "auroc_stored": auroc(m.ptx, m.logit_stored) if ds != "ptx498" else float("nan")})
                df.to_csv(a.out / "topk" / f"{tag}__{ds}.csv.gz", index=False)
                if seg:
                    H = np.stack(hist)
                    for gname, sel in {"pos_dev": (df.ptx == 1) & (df.treated == 1), "pos_nodev": (df.ptx == 1) & (df.treated == 0),
                                       "neg_dev": (df.ptx == 0) & (df.treated == 1), "neg_nodev": (df.ptx == 0) & (df.treated == 0),
                                       "pos": df.ptx == 1}.items():
                        if sel.sum():
                            key = f"{run}|{ds}|{gname}"
                            maps[key] = maps.get(key, 0) + H[sel.values].sum(0)
                            maps[key + "|n"] = maps.get(key + "|n", 0) + int(sel.sum())
                if ds in ("siim", "nih"):
                    y = df.treated.values.astype(int)
                    res = {"device": probe_auc(feats, y, df.patient_id.values)}
                    if ds in drain:
                        yd = np.array([drain[ds].get(i, -1) for i in df.image_id])
                        res["drain"] = probe_auc(feats, yd, df.patient_id.values)
                        res["drain_n_labelled"] = int((yd >= 0).sum())
                    res["ptx"] = probe_auc(feats, df.ptx.values.astype(int), df.patient_id.values)
                    probes[f"{tag}|{ds}"] = res
            if dev.type == "cuda" and sd.name == "seed0":
                try:
                    comp.append({"run": run, **complexity(model, cfg, dev)})
                except Exception as e:  # noqa: BLE001
                    comp.append({"run": run, "error": str(e)[:200]})
                pd.DataFrame(comp).to_csv(a.out / f"complexity_{run}.csv", index=False)
                comp = []
            del model
            torch.cuda.empty_cache()
            pd.DataFrame(repro).to_csv(a.out / f"repro_{tag}.csv", index=False); repro = []
            json.dump(probes, open(a.out / "probe.json", "w"), indent=1)
            if maps:
                old = dict(np.load(a.out / "topk_maps.npz")) if (a.out / "topk_maps.npz").exists() else {}
                for k, v in maps.items():
                    old[k] = old.get(k, 0) + np.asarray(v)
                np.savez_compressed(a.out / "topk_maps.npz", **{k: np.asarray(v) for k, v in old.items()})
                maps = {}
            (a.out / f"DONE_{tag}").write_text("ok")
            print("done", tag, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
