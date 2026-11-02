"""Summarise the GPU outputs of scripts/r1_gpu.py (results/r1/, pushed by the pod to branch r1-results)
together with the stored predictions into results/r1/gpu_summary.json.

    PYTHONPATH=. python scripts/r1_summarize_gpu.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptx.eval.acceptance import auc_sub, load_run
from ptx.eval.stats import cluster_bootstrap

R1 = Path("results/r1")
RUNS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_sens_lambda001", "pb_sens_lambda1",
        "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls",
        "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
J = {}
r3 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)  # noqa: E731

# reproducibility of stored logits from the released weights
rep = [pd.read_csv(f) for f in glob.glob(str(R1 / "repro_*.csv")) if Path(f).stat().st_size > 5]
if rep:
    rep = pd.concat(rep)
    J["repro"] = {"n_run_seed_sets": int(len(rep)), "max_abs_logit_diff": r3(rep.max_abs_logit_diff.max()),
                  "median_max_abs_logit_diff": r3(rep.max_abs_logit_diff.median()),
                  "max_abs_auroc_diff": r3((rep.auroc_now - rep.auroc_stored).abs().max()),
                  "per_dataset_max": rep.groupby("dataset").max_abs_logit_diff.max().round(4).to_dict()}

# top-k overlap with the reference mask (positives with a mask), averaged over images then seeds
tk = {}
for r in RUNS:
    for ds in ("siim", "ptx498"):
        fs = sorted(glob.glob(str(R1 / "topk" / f"{r}__seed*__{ds}.csv.gz")))
        if not fs:
            continue
        per = []
        for f in fs:
            d = pd.read_csv(f)
            d = d[(d.ptx == 1) & d.topk_in_gt.notna()]
            per.append({k: d[k].mean() for k in ("topk_in_gt", "topk_in_gt_dil", "argmax_in_gt", "argmax_in_gt_dil",
                                                    "topk_any_in_gt", "chance_in_gt", "chance_in_gt_dil")} | {"n": len(d)})
            if ds == "siim":  # by device stratum (pseudo-labels) and by size tertile
                for g, sel in [("dev", d.treated == 1), ("nodev", d.treated == 0)]:
                    per[-1][f"topk_in_gt_{g}"] = d[sel].topk_in_gt.mean()
                    per[-1][f"topk_in_gt_dil_{g}"] = d[sel].topk_in_gt_dil.mean()
                q = np.quantile(d.gt_area_frac, [1 / 3, 2 / 3]); t = np.digitize(d.gt_area_frac, q)
                for k in range(3):
                    per[-1][f"topk_in_gt_dil_t{k}"] = d[t == k].topk_in_gt_dil.mean()
        P = pd.DataFrame(per)
        tk.setdefault(r, {})[ds] = {k: r3(P[k].mean()) for k in P.columns} | {"seeds": len(fs), "seed_sd_topk_in_gt": r3(P.topk_in_gt.std())}
J["topk"] = tk

# false positives: where do the top-k pixels of high-scoring NIH negatives sit? (centroids, device vs no device)
fp = {}
for r in ["pb_full", "pb_no_c2", "pb_base", "pb_bl_segformer_b2", "pb_bl_groupdro"]:
    fs = sorted(glob.glob(str(R1 / "topk" / f"{r}__seed*__nih.csv.gz")))
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs])
        g = d.groupby("image_id").agg(ptx=("ptx", "first"), treated=("treated", "first"), logit=("logit", "mean"),
                                      spread=("topk_spread", "mean"), cy=("topk_cy", "mean"))
        neg = g[g.ptx == 0]
        fp[r] = {"neg_dev_spread": r3(neg[neg.treated == 1].spread.median()), "neg_nodev_spread": r3(neg[neg.treated == 0].spread.median())}
J["nih_neg_topk"] = fp

# device probes
pj = R1 / "probe.json"
if pj.exists():
    P = json.load(open(pj))
    out = {}
    for r in RUNS:
        for ds in ("nih", "siim"):
            keys = [k for k in P if k.startswith(r + "__") and k.endswith("|" + ds) and P[k].get("device")]
            if not keys:
                continue
            o = {}
            for tgt in ("device", "drain", "ptx"):
                for cv in ("image_cv", "image_cv_std", "patient_cv"):
                    v = [P[k][tgt][cv] for k in keys if P[k].get(tgt) and P[k][tgt].get(cv) is not None]
                    if v:
                        o[f"{tgt}_{cv}"] = r3(np.mean(v))
                if P[keys[0]].get(tgt):
                    o[f"{tgt}_n"] = P[keys[0]][tgt]["n"]; o[f"{tgt}_npos"] = P[keys[0]][tgt]["n_pos"]
            out.setdefault(r, {})[ds] = o | {"seeds": len(keys)}
    J["probe"] = out

# complexity
cs = [pd.read_csv(f) for f in glob.glob(str(R1 / "complexity_*.csv"))]
if cs:
    c = pd.concat(cs).set_index("run")
    tt = pd.read_csv(R1 / "train_time.csv").set_index("run") if (R1 / "train_time.csv").exists() else None
    if tt is not None:
        c = c.join(tt[["total_min", "min_per_epoch"]])
    J["complexity"] = json.loads(c.round(3).to_json(orient="index"))

# NIH metadata strata (view position, sex, age) for the device--device-absent gap
mp = R1 / "nih_metadata.csv"
if mp.exists():
    m = pd.read_csv(mp)
    meta = m.set_index("image_id")
    strata = {}
    for r in RUNS:
        L = load_run(Path("results/runs"), r, "siim", "nih", "test")
        d = L["df"].join(meta[["View Position", "Patient Gender", "Patient Age"]], on="image_id")
        o = {}
        for col, groups in [("View Position", ["PA", "AP"]), ("Patient Gender", ["M", "F"])]:
            for gv in groups:
                dd = d[d[col] == gv].reset_index(drop=True)
                n_u = int(((dd.ptx == 1) & (dd.treated == 0)).sum()); n_d = int(((dd.ptx == 1) & (dd.treated == 1)).sum())
                o[f"{col}={gv}"] = {"n_dev": n_d, "n_nodev": n_u, "n_neg": int((dd.ptx == 0).sum()),
                                     "auc_dev": r3(auc_sub(dd, "treated")) if n_d >= 5 else None,
                                     "auc_nodev": r3(auc_sub(dd, "untreated")) if n_u >= 5 else None}
                if o[f"{col}={gv}"]["auc_dev"] is not None and o[f"{col}={gv}"]["auc_nodev"] is not None:
                    o[f"{col}={gv}"]["gap"] = r3(o[f"{col}={gv}"]["auc_dev"] - o[f"{col}={gv}"]["auc_nodev"])
        # adjusted: logistic model of the score rank among positives on device + view + sex + age
        import statsmodels.formula.api as smf
        pos = d[(d.ptx == 1) & (d.treated >= 0)].copy()
        pos["z"] = np.log(np.clip(pos.score, 1e-6, 1 - 1e-6) / (1 - np.clip(pos.score, 1e-6, 1 - 1e-6)))
        pos = pos.rename(columns={"View Position": "view", "Patient Gender": "sex", "Patient Age": "age"})
        fit = smf.ols("z ~ treated + C(view) + C(sex) + age", data=pos).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pos.patient_id)[0]})
        un = smf.ols("z ~ treated", data=pos).fit()
        o["logit_on_device_adjusted"] = {"coef": r3(fit.params["treated"]), "lo": r3(fit.conf_int().loc["treated", 0]),
                                         "hi": r3(fit.conf_int().loc["treated", 1]), "p": r3(fit.pvalues["treated"]),
                                         "coef_unadj": r3(un.params["treated"])}
        strata[r] = o
    t = load_run(Path("results/runs"), "pb_full", "siim", "nih", "test")["df"].join(meta[["View Position", "Patient Gender", "Patient Age"]], on="image_id")
    comp = t.groupby(["ptx", "treated"]).agg(n=("image_id", "size"), ap=("View Position", lambda s: (s == "AP").mean()),
                                              male=("Patient Gender", lambda s: (s == "M").mean()), age=("Patient Age", "median"))
    J["nih_metadata_composition"] = {f"ptx{k[0]}_dev{k[1]}": {kk: r3(vv) for kk, vv in v.items()} for k, v in comp.to_dict(orient="index").items()}
    J["nih_metadata_strata"] = strata

# image statistics (threshold transfer)
a, b = R1 / "siim_image_stats.csv", R1 / "nih_image_stats.csv"
if a.exists() and b.exists():
    A, B = pd.read_csv(a), pd.read_csv(b)
    J["image_stats"] = {c: {"siim_median": r3(A[c].median()), "nih_median": r3(B[c].median())} for c in A.columns if c != "image_id"}
json.dump(J, open(R1 / "gpu_summary.json", "w"), indent=1)
print(json.dumps(J, indent=1)[:3000])
