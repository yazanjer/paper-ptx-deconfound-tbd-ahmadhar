"""Revision R1 (IEEE Access resubmission): analyses computed from the stored per-image predictions only.

Nothing is retrained or re-tuned; thresholds are the development-validation thresholds used in the paper.
Writes results/r1/cpu.json (all numbers quoted in the revision) and a few CSV tables.

    PYTHONPATH=. python scripts/r1_cpu.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression

from ptx.eval.acceptance import auc_sub, load_run, masks, sens_untreated
from ptx.eval.metrics import auroc
from ptx.eval.stats import boot_p_two_sided, cluster_bootstrap, holm

R = Path("results/runs")
OUT = Path("results/r1"); OUT.mkdir(parents=True, exist_ok=True)
INP = Path("results/r1_inputs")
PROP = "pb_full"
RUNS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_sens_lambda001", "pb_sens_lambda1",
        "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls",
        "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
BASELINES = ["pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls",
             "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
B = 2000
J = {}


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def ci(point, lo, hi):
    return [r3(point), r3(lo), r3(hi)]


L = {ds: {r: load_run(R, r, "siim", ds, sp) for r in RUNS}
     for ds, sp in [("nih", "test"), ("siim", "test_internal"), ("ptx498", "test")]}
VAL = {r: L["nih"][r]["val"] for r in RUNS}

# ---------------------------------------------------------------- 0. reproduce the headline numbers
rep = {}
for r in RUNS:
    d, t = L["nih"][r]["df"], L["nih"][r]["thr"]
    rep[r] = {"auroc_untr": r3(auc_sub(d, "untreated")), "auroc_dev": r3(auc_sub(d, "treated")),
              "gap": r3(auc_sub(d, "treated") - auc_sub(d, "untreated")), "sens_untr": r3(sens_untreated(d, t)),
              "n_untr_detected": int(((d.ptx == 1) & (d.treated == 0) & (d.score >= t)).sum()),
              "n_untr": int(((d.ptx == 1) & (d.treated == 0)).sum()), "thr": r3(t)}
J["nih_reproduced"] = rep

# ---------------------------------------------------------------- 1. paired comparisons, proposed vs each comparator (R1-4, R2-17)
paired = {}
P = L["nih"][PROP]
for r in RUNS[1:]:
    C = L["nih"][r]
    a, b = P["df"].set_index("image_id"), C["df"].set_index("image_id").loc[P["df"].image_id]
    a = a.reset_index(); b = b.reset_index()
    cl = a.patient_id.values

    def f_auc(idx):
        return auc_sub(a, "untreated", idx) - auc_sub(b, "untreated", idx)

    def f_gap(idx):
        return (auc_sub(a, "treated", idx) - auc_sub(a, "untreated", idx)) - (auc_sub(b, "treated", idx) - auc_sub(b, "untreated", idx))

    def f_sens(idx):
        return sens_untreated(a, P["thr"], idx) - sens_untreated(b, C["thr"], idx)

    res = {}
    for name, fn in [("d_auc_untr", f_auc), ("d_gap", f_gap), ("d_sens_untr", f_sens)]:
        pt, lo, hi, s = cluster_bootstrap(fn, cl, B, seed=0)
        res[name] = ci(pt, lo, hi) + [r3(boot_p_two_sided(s))]
    unt = ((a.ptx == 1) & (a.treated == 0)).values
    da, db = a.score.values[unt] >= P["thr"], b.score.values[unt] >= C["thr"]
    res["untr_detected"] = [int(da.sum()), int(db.sum()), int(unt.sum())]
    res["discordant_prop_only"], res["discordant_comp_only"] = int((da & ~db).sum()), int((~da & db).sum())
    paired[r] = res
for key in ("d_auc_untr", "d_gap", "d_sens_untr"):
    ps = [paired[r][key][3] for r in RUNS[1:]]
    for r, p in zip(RUNS[1:], holm(ps)):
        paired[r][key].append(r3(p))
J["paired_nih"] = paired  # each: [point, lo, hi, p_boot, p_holm]

# ---------------------------------------------------------------- 2. threshold transfer (R1-7, R2-16)
tt = {}
for r in RUNS:
    v, n, s = L["nih"][r]["val"], L["nih"][r]["df"], L["siim"][r]["df"]
    t = L["nih"][r]["thr"]
    vn, nn, sn = v[v.ptx == 0].score, n[n.ptx == 0].score, s[s.ptx == 0].score
    row = {"spec_siim_val": r3((vn < t).mean()), "spec_siim_test": r3((sn < t).mean()), "spec_nih": r3((nn < t).mean()),
           "spec_nih_negdev": r3((n[(n.ptx == 0) & (n.treated == 1)].score < t).mean()),
           "spec_nih_negnodev": r3((n[(n.ptx == 0) & (n.treated == 0)].score < t).mean()),
           "neg_q50_siim_val": r3(vn.median()), "neg_q50_nih": r3(nn.median()),
           "neg_q95_siim_val": r3(vn.quantile(.95)), "neg_q95_nih": r3(nn.quantile(.95)),
           "neg_q99_siim_val": r3(vn.quantile(.99)), "neg_q99_nih": r3(nn.quantile(.99)),
           "ks_neg_siimval_vs_nih": r3(ks_2samp(v[v.ptx == 0].logit, n[n.ptx == 0].logit).statistic),
           "pos_q50_siim_test": r3(s[s.ptx == 1].score.median()), "pos_q50_nih": r3(n[n.ptx == 1].score.median()),
           "pos_untr_q50_siim_test": r3(s[(s.ptx == 1) & (s.treated == 0)].score.median()),
           "pos_untr_q50_nih": r3(n[(n.ptx == 1) & (n.treated == 0)].score.median()),
           "nih_thr_for_spec95": r3(np.quantile(nn, .95)), "thr": r3(t)}
    nt = np.quantile(nn, .95)  # oracle (NIH-recalibrated) threshold, exploratory
    row["oracle_sens_untr"] = r3((n[(n.ptx == 1) & (n.treated == 0)].score >= nt).mean())
    row["oracle_sens_dev"] = r3((n[(n.ptx == 1) & (n.treated == 1)].score >= nt).mean())
    tt[r] = row
J["threshold_transfer"] = tt
vneg = L["nih"][PROP]["val"]
siim_val_lab = pd.read_csv("results/device_labels/siim_device_labels.csv", dtype={"image_id": str})
sv = pd.read_csv(sorted(R.glob(f"{PROP}/seed0/preds_siim_val.csv"))[0], dtype={"image_id": str})
nn = L["nih"][PROP]["df"]
J["prevalence"] = {
    "siim_val": {"n": len(sv), "ptx": r3(sv.ptx.mean()), "device_neg": r3(sv[sv.ptx == 0].treated.mean()),
                 "device_pos": r3(sv[sv.ptx == 1].treated.mean())},
    "nih_test": {"n": len(nn), "ptx": r3(nn.ptx.mean()), "device_neg": r3(nn[(nn.ptx == 0) & (nn.treated >= 0)].treated.mean()),
                 "device_pos": r3(nn[(nn.ptx == 1) & (nn.treated >= 0)].treated.mean())}}

# ---------------------------------------------------------------- 3. severity: size-stratified gap on SIIM internal test (R1-1, R2-14)
sev = {}
AREA = L["siim"][PROP]["df"].set_index("image_id").gt_area_frac  # property of the image (reference mask), not of the model
for r in RUNS:
    d = L["siim"][r]["df"].copy()
    d["gt_area_frac"] = d.image_id.map(AREA)
    pos = d[d.ptx == 1]
    edges = np.quantile(pos.gt_area_frac, [1 / 3, 2 / 3])
    d["tert"] = np.where(d.ptx == 1, np.digitize(d.gt_area_frac.fillna(0), edges), -1)
    out = {"edges_area_frac": [r3(e) for e in edges]}
    for k in range(3):
        m = (d.tert == k) | (d.ptx == 0)
        dd = d[m]
        _, _, unt, trt = masks(dd)
        out[f"t{k}"] = {"n_dev": int(trt.sum()), "n_nodev": int(unt.sum()),
                        "auc_dev": r3(auc_sub(dd, "treated")), "auc_nodev": r3(auc_sub(dd, "untreated")),
                        "gap": r3(auc_sub(dd, "treated") - auc_sub(dd, "untreated"))}
    gaps = [out[f"t{k}"]["gap"] for k in range(3)]
    out["gap_unadjusted"] = r3(auc_sub(d, "treated") - auc_sub(d, "untreated"))
    out["gap_size_stratified_mean"] = r3(np.nanmean(gaps))
    p = pos[pos.treated >= 0]
    X = np.c_[p.treated.values, np.log(p.gt_area_frac.values + 1e-5)]
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X, (p.score > L["siim"][r]["thr"]).astype(int)) if p.treated.nunique() == 2 else None
    out["pos_dev_vs_area_mean"] = {"area_dev": r3(p[p.treated == 1].gt_area_frac.median()),
                                   "area_nodev": r3(p[p.treated == 0].gt_area_frac.median())}
    from statsmodels.formula.api import ols
    p2 = p.assign(lg=np.log(p.gt_area_frac + 1e-5), z=p.logit)
    fit = ols("z ~ treated + lg", data=p2).fit()
    out["logit_on_device_adj_size"] = {"coef": r3(fit.params["treated"]), "lo": r3(fit.conf_int().loc["treated", 0]),
                                       "hi": r3(fit.conf_int().loc["treated", 1]), "p": r3(fit.pvalues["treated"]),
                                       "coef_unadj": r3(ols("z ~ treated", data=p2).fit().params["treated"])}
    sev[r] = out
J["severity_siim_test"] = sev

# ---------------------------------------------------------------- 4. pseudo-label validity vs NEATX chest-drain labels (R1-2, R2-4)
dr = pd.read_csv(INP / "siim_drain_labels.csv", dtype={"image_id": str})
lab = siim_val_lab.merge(dr[["image_id", "split", "ptx", "Drain"]], on="image_id")
pl = {}
for name, sub in [("all", lab), ("ptx_pos", lab[lab.ptx == 1]), ("ptx_neg", lab[lab.ptx == 0]),
                  ("test_internal", lab[lab.split == "test_internal"])]:
    ct = pd.crosstab(sub.Drain.astype(int), sub.device.astype(int))
    g = lambda i, j: int(ct.loc[i, j]) if i in ct.index and j in ct.columns else 0  # noqa: E731
    pl[name] = {"n": len(sub), "drain1_dev1": g(1, 1), "drain1_dev0": g(1, 0), "drain0_dev1": g(0, 1), "drain0_dev0": g(0, 0),
                "sens_pseudo_for_drain": r3(g(1, 1) / max(1, g(1, 1) + g(1, 0))),
                "drain_share_among_pseudo_dev": r3(g(1, 1) / max(1, g(1, 1) + g(0, 1))),
                "nodrain_but_device": r3(g(0, 1) / max(1, g(0, 1) + g(0, 0))),
                "auc_device_prob_for_drain": r3(auroc(sub.Drain.astype(int), sub.device_prob))}
J["pseudo_vs_drain_siim"] = pl

# ---------------------------------------------------------------- 5. chest-drain-specific strata (R1-2, R2-4)
drs = {}
dmap = dict(zip(dr.image_id, dr.Drain.astype(int)))
nd = pd.read_csv(INP / "nih_neatx_drain.csv", dtype={"image_id": str})
nmap = dict(zip(nd.image_id, nd.Drain.astype(int)))
for r in RUNS:
    out = {}
    d = L["siim"][r]["df"].copy()
    d["drain"] = d.image_id.map(dmap).fillna(-1).astype(int)
    m = d.drain >= 0
    sub = d[m].copy()
    neg = sub.ptx == 0
    for nm, sel in [("pos_drain", (sub.ptx == 1) & (sub.drain == 1)), ("pos_nodrain", (sub.ptx == 1) & (sub.drain == 0))]:
        mm = sel | neg
        out[f"siim_auc_{nm}"] = r3(auroc(sub.ptx[mm], sub.score[mm]))
        out[f"siim_n_{nm}"] = int(sel.sum())
    out["siim_gap_drain"] = r3(out["siim_auc_pos_drain"] - out["siim_auc_pos_nodrain"])
    sub2 = sub.copy(); sub2["treated"] = sub2.drain
    pt, lo, hi, _ = cluster_bootstrap(lambda idx: auc_sub(sub2, "treated", idx) - auc_sub(sub2, "untreated", idx), sub2.image_id.values, B)
    out["siim_gap_drain_ci"] = ci(pt, lo, hi)
    pt, lo, hi, _ = cluster_bootstrap(lambda idx: auc_sub(sub, "treated", idx) - auc_sub(sub, "untreated", idx), sub.image_id.values, B)
    out["siim_gap_pseudodev_same_subset_ci"] = ci(pt, lo, hi)
    ng = sub[neg]
    out["siim_neg_drain_in_score_auc"] = r3(auroc(ng.drain, ng.score))  # among negatives: does a drain raise the score?
    out["siim_neg_n_drain"] = int(ng.drain.sum()); out["siim_neg_n"] = len(ng)
    n = L["nih"][r]["df"].copy()
    n["drain"] = n.image_id.map(nmap).fillna(-1).astype(int)
    negs = n[n.ptx == 0]
    for nm, sel in [("pos_drain", (n.ptx == 1) & (n.drain == 1)), ("pos_nodrain_neatx", (n.ptx == 1) & (n.drain == 0)),
                    ("pos_device_nodrain", (n.ptx == 1) & (n.drain == 0) & (n.treated == 1))]:
        mm = sel | (n.ptx == 0)
        out[f"nih_auc_{nm}"] = r3(auroc(n.ptx[mm], n.score[mm])) if sel.sum() >= 5 else None
        out[f"nih_n_{nm}"] = int(sel.sum())
    drs[r] = out
J["drain_strata"] = drs
n = L["nih"][PROP]["df"]
n["drain"] = n.image_id.map(nmap).fillna(-1).astype(int)
J["nih_human_device_vs_neatx_drain"] = pd.crosstab(n[n.drain >= 0].drain, n[n.drain >= 0].treated).to_dict()

# ---------------------------------------------------------------- 6. patient overlap through NEATX-mapped images (R1-3, R2-6)
pairs = pd.read_csv("../paper-ptx-drain-invariance-tbd-ahmadhar/data/siim_neatx_pairs.csv", dtype={"image_id": str, "nih_id": str}) \
    if Path("../paper-ptx-drain-invariance-tbd-ahmadhar/data/siim_neatx_pairs.csv").exists() else None
if pairs is None and (INP / "siim_neatx_pairs.csv").exists():
    pairs = pd.read_csv(INP / "siim_neatx_pairs.csv", dtype={"image_id": str, "nih_id": str})
if pairs is not None:
    pairs.to_csv(INP / "siim_neatx_pairs.csv", index=False)
    sp = pd.read_csv("splits/siim.csv", dtype={"image_id": str})
    pm = pairs.drop_duplicates(["image_id", "nih_id"]).merge(sp, on="image_id")
    pm["nih_patient"] = pm.nih_id.str[:8]
    npat = pm.groupby("image_id").nih_patient.nunique()
    J["patient_overlap_mapping"] = {"siim_images_matched": int(len(npat)), "ambiguous_multi_patient": int((npat > 1).sum()),
                                    "hamming_counts": pairs.drop_duplicates(["image_id", "nih_id"]).hamming.astype(int).value_counts().sort_index().to_dict()}
    pm = pm[pm.image_id.isin(npat[npat == 1].index)].drop_duplicates("image_id")  # one unambiguous NIH patient per SIIM image
    per_pat = pm.groupby("nih_patient").split.nunique()
    multi = per_pat[per_pat > 1].index
    tst = pm[pm.split == "test_internal"]
    trpat = set(pm[pm.split.isin(["train", "val"])].nih_patient)
    nih = pd.read_csv("splits/nih.csv", dtype={"image_id": str})
    nih_test_pat = set(nih[nih.split == "test"].image_id.str[:8])
    J["patient_overlap_mapped_subset"] = {
        "siim_images_mapped": int(pm.image_id.nunique()), "nih_patients": int(pm.nih_patient.nunique()),
        "patients_in_more_than_one_siim_split": int(len(multi)),
        "siim_test_images_mapped": int(tst.image_id.nunique()),
        "siim_test_images_whose_patient_in_train_or_val": int(tst[tst.nih_patient.isin(trpat)].image_id.nunique()),
        "nih_test_patients": len(nih_test_pat),
        "nih_test_patients_with_mapped_siim_image": len(nih_test_pat & set(pm.nih_patient)),
        "nih_test_images_of_those_patients": int(nih[(nih.split == "test") & nih.image_id.str[:8].isin(set(pm.nih_patient))].shape[0]),
        "mapped_siim_images_from_nih_test_patients_by_split": pm[pm.nih_patient.isin(nih_test_pat)].groupby("split").image_id.nunique().to_dict()}
    # sensitivity analysis: drop NIH test images whose patient has a mapped SIIM development image
    dev_pat = set(pm[pm.split.isin(["train", "val"])].nih_patient)
    keep = ~n.image_id.str[:8].isin(dev_pat)
    J["patient_overlap_mapped_subset"]["nih_test_dropped"] = int((~keep).sum())
    sens_rows = {}
    for r in RUNS:
        d = L["nih"][r]["df"]; k = ~d.image_id.str[:8].isin(dev_pat)
        dd = d[k].reset_index(drop=True)
        _, _, unt, trt = masks(dd)
        row = {"n": len(dd), "n_dev_pos": int(trt.sum()), "n_nodev_pos": int(unt.sum()), "n_neg": int((dd.ptx == 0).sum()),
               "sens_untr": r3(sens_untreated(dd, L["nih"][r]["thr"]))}
        for nm, fn in [("auc_untr", lambda idx, dd=dd: auc_sub(dd, "untreated", idx)),
                       ("auc_dev", lambda idx, dd=dd: auc_sub(dd, "treated", idx)),
                       ("gap", lambda idx, dd=dd: auc_sub(dd, "treated", idx) - auc_sub(dd, "untreated", idx))]:
            pt, lo, hi, _ = cluster_bootstrap(fn, dd.patient_id.values, B)
            row[nm] = ci(pt, lo, hi)
        sens_rows[r] = row
    J["patient_overlap_mapped_subset"]["nih_metrics_after_exclusion"] = sens_rows
    dd = L["nih"][PROP]["df"]; dropped = dd[dd.image_id.str[:8].isin(dev_pat)]
    J["patient_overlap_mapped_subset"]["dropped_composition"] = {
        "pos_dev": int(((dropped.ptx == 1) & (dropped.treated == 1)).sum()),
        "pos_nodev": int(((dropped.ptx == 1) & (dropped.treated == 0)).sum()), "neg": int((dropped.ptx == 0).sum())}

# ---------------------------------------------------------------- 7. shared behaviour across models (R2-18)
S = pd.DataFrame({r: L["nih"][r]["df"].set_index("image_id").score for r in RUNS})
meta = L["nih"][PROP]["df"].set_index("image_id")
corr = {}
for nm, sel in [("untr_pos", (meta.ptx == 1) & (meta.treated == 0)), ("neg", meta.ptx == 0), ("all", meta.ptx >= 0)]:
    c = S[sel.values].rank().corr(method="pearson")
    vals = [c.loc[a, b] for a, b in combinations(RUNS, 2)]
    corr[nm] = {"median_spearman": r3(np.median(vals)), "min": r3(np.min(vals)), "max": r3(np.max(vals))}
missed = ((S[((meta.ptx == 1) & (meta.treated == 0)).values] >= pd.Series({r: L["nih"][r]["thr"] for r in RUNS})).sum(1))
J["shared_behaviour"] = {"score_rank_correlation": corr,
                         "untr_pos_detected_by_k_models": missed.value_counts().sort_index().to_dict()}

# ---------------------------------------------------------------- 8. hyper-parameter sensitivity summary (R2-7, R2-8)
hp = {}
for r, lam in [("pb_sens_lambda001", 0.01), ("pb_full", 0.1), ("pb_sens_lambda1", 1.0), ("pb_no_c2", 0.0)]:
    v = L["nih"][r]["val"]
    hp[r] = {"lambda": lam, "val_auroc": r3(auroc(v.ptx, v.score)), **rep[r],
             "siim_test_auroc": r3(auroc(L["siim"][r]["df"].ptx, L["siim"][r]["df"].score))}
J["lambda_sensitivity"] = hp

json.dump(J, open(OUT / "cpu.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: J[k] for k in ("pseudo_vs_drain_siim", "patient_overlap_mapped_subset")}, indent=1, default=str)[:4000])
