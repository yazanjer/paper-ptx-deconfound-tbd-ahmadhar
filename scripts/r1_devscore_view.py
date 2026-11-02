"""Revision R1: device-in-score AUROC among NIH negatives stratified by projection (PA/AP), from stored predictions
and the ChestX-ray14 metadata (results/r1/nih_metadata.csv). Writes results/r1/devscore_by_view.json.

    PYTHONPATH=. python scripts/r1_devscore_view.py
"""
import json
from pathlib import Path

import pandas as pd

from ptx.eval.acceptance import load_run
from ptx.eval.metrics import auroc
from ptx.eval.stats import cluster_bootstrap

m = pd.read_csv("results/r1/nih_metadata.csv").set_index("image_id")
RUNS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_sens_lambda001", "pb_sens_lambda1", "pb_bl_unet_r34_kaggle",
        "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls", "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
out = {}
for r in RUNS:
    d = load_run(Path("results/runs"), r, "siim", "nih", "test")["df"].join(m[["View Position"]], on="image_id")
    n = d[(d.ptx == 0) & (d.treated >= 0)]
    pa = n[n["View Position"] == "PA"].reset_index(drop=True)
    ap = n[n["View Position"] == "AP"].reset_index(drop=True)
    p, lo, hi, _ = cluster_bootstrap(lambda idx: auroc(pa.treated.values[idx], pa.score.values[idx]), pa.patient_id.values, 2000)
    out[r] = dict(all=round(auroc(n.treated, n.score), 3), PA=[round(p, 3), round(lo, 3), round(hi, 3)],
                  nPA=(int(pa.treated.sum()), int((pa.treated == 0).sum())), AP=round(auroc(ap.treated, ap.score), 3),
                  nAP=(int(ap.treated.sum()), int((ap.treated == 0).sum())))
json.dump(out, open("results/r1/devscore_by_view.json", "w"), indent=1)
print(pd.DataFrame(out).T)
