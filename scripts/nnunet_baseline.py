"""nnU-Net v2 (2D) baseline: export -> train with nnU-Net's own CLI -> import predictions.

    python scripts/nnunet_baseline.py export --data /workspace/data --nnraw $nnUNet_raw
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    nnUNetv2_train 501 2d 0                       # fold 0 == our frozen train/val split
    nnUNetv2_predict -i $nnUNet_raw/Dataset501_PTX/imagesTs_<set> -o pred_<set> -d 501 -c 2d -f 0 --save_probabilities
    python scripts/nnunet_baseline.py import --data /workspace/data --pred-root . --seed 0
Repeat train/predict/import with --seed 1, 2 for three (non-deterministic) repeats.
The image-level score is the top-0.5% mean foreground probability, as for our models (C4).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ptx.data.dataset import load_table
from ptx.eval.metrics import dice, threshold_at_specificity
from ptx.eval.size import size_features

DS = "Dataset501_PTX"
SETS = [("candid", "val"), ("candid", "test_internal"), ("siim", "test"), ("nih", "test"), ("ptx498", "test")]


def export(a):
    root = Path(a.nnraw) / DS
    for d in ("imagesTr", "labelsTr"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tr = load_table(a.data, "candid", a.splits, ["train", "val"])
    for r in tr.itertuples():
        img = cv2.imread(str(a.data / "candid" / "img" / f"{r.image_id}.png"), 0)
        m = cv2.imread(str(a.data / "candid" / "ptx" / f"{r.image_id}.png"), 0)
        cv2.imwrite(str(root / "imagesTr" / f"{r.image_id}_0000.png"), img)
        cv2.imwrite(str(root / "labelsTr" / f"{r.image_id}.png"), (m > 0).astype(np.uint8))
    json.dump({"channel_names": {"0": "xray"}, "labels": {"background": 0, "ptx": 1},
               "numTraining": len(tr), "file_ending": ".png"}, open(root / "dataset.json", "w"), indent=2)
    folds = [{"train": tr.loc[tr.split == "train", "image_id"].tolist(), "val": tr.loc[tr.split == "val", "image_id"].tolist()}]
    Path(a.nnraw).parent.joinpath("nnUNet_preprocessed", DS).mkdir(parents=True, exist_ok=True)
    json.dump(folds, open(Path(a.nnraw).parent / "nnUNet_preprocessed" / DS / "splits_final.json", "w"))
    for ds, sp in SETS:
        try:
            t = load_table(a.data, ds, a.splits, sp)
        except FileNotFoundError:
            continue
        d = root / f"imagesTs_{ds}_{sp}"
        d.mkdir(exist_ok=True)
        for r in t.itertuples():
            cv2.imwrite(str(d / f"{r.image_id}_0000.png"), cv2.imread(str(a.data / ds / "img" / f"{r.image_id}.png"), 0))
    print("exported; copy splits_final.json into nnUNet_preprocessed after planning if it gets overwritten")


def import_(a):
    od = Path("results/runs/bl_nnunet") / f"seed{a.seed}"
    od.mkdir(parents=True, exist_ok=True)
    for ds, sp in SETS:
        pdir = Path(a.pred_root) / f"pred_{ds}_{sp}"
        if not pdir.exists():
            continue
        t = load_table(a.data, ds, a.splits, sp)
        rows = []
        for r in t.itertuples():
            prob = np.load(pdir / f"{r.image_id}.npz")["probabilities"][1].squeeze()
            flat = np.sort(prob.ravel())[::-1]
            s = float(np.clip(flat[: max(1, int(0.005 * flat.size))].mean(), 1e-6, 1 - 1e-6))
            pm = prob > 0.5
            row = {"image_id": r.image_id, "dataset": ds, "patient_id": r.patient_id, "ptx": r.ptx, "treated": r.treated,
                   "treated_source": r.treated_source, "score": s, "logit": math.log(s / (1 - s))}
            row.update({f"pred_{k}": v for k, v in size_features(pm, r.spacing_mm).items()})
            if r.has_ptx_mask == 1:
                gm = cv2.resize(cv2.imread(str(a.data / ds / "ptx" / f"{r.image_id}.png"), 0), pm.shape[::-1]) > 0
                row.update({f"gt_{k}": v for k, v in size_features(gm, r.spacing_mm).items()})
                row["dice"] = dice(pm, gm)
            rows.append(row)
        df = pd.DataFrame(rows)
        name = f"preds_{ds}_{sp}.csv"
        df.to_csv(od / name, index=False)
        if (ds, sp) == ("candid", "val"):
            json.dump({"threshold_spec95": threshold_at_specificity(df.ptx, df.score, 0.95)}, open(od / "threshold.json", "w"))
    (od / "DONE").write_text("nnunet import")
    print("imported to", od)


ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["export", "import"])
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--splits", type=Path, default=Path("splits"))
ap.add_argument("--nnraw", default="/workspace/nnUNet_raw")
ap.add_argument("--pred-root", default=".")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
export(a) if a.cmd == "export" else import_(a)
