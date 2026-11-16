"""Second revision (resubmission of Access-2026-56789): analyses computed from stored predictions and logs only.

No model is retrained, re-selected or re-thresholded. Thresholds are the SIIM-ACR validation thresholds of the paper.
Writes results/r2/cpu.json.

    PYTHONPATH=. python scripts/r2_cpu.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr

from ptx.eval.acceptance import load_run
from ptx.eval.metrics import threshold_at_specificity
from ptx.eval.stats import boot_p_two_sided, delong_test, holm

R = Path("results/runs")
OUT = Path("results/r2"); OUT.mkdir(parents=True, exist_ok=True)
INP = Path("results/r1_inputs")
PROP = "pb_full"
RUNS = ["pb_full", "pb_no_c2", "pb_no_c4", "pb_base", "pb_sens_lambda001", "pb_sens_lambda1",
        "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls",
        "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
PRESPEC_BL = ["pb_no_c4", "pb_bl_unet_r34_kaggle", "pb_bl_segformer_b2", "pb_bl_groupdro", "pb_bl_bandaru_auxcls",
              "pb_bl_raddino_probe", "pb_bl_raddino_ft"]
B = 2000
J: dict = {}


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [r4(max(0.0, c - h)), r4(min(1.0, c + h))]


def auc(pos, neg):
    """Mann-Whitney AUROC, ties count one half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ------------------------------------------------------------------ load everything once, aligned on the NIH image order
L = {r: load_run(R, r, "siim", "nih", "test") for r in RUNS}
META = L[PROP]["df"][["image_id", "patient_id", "ptx", "treated"]].reset_index(drop=True)
IDS = META.image_id.values
Y = META.ptx.values
DEV = META.treated.values
PAT = META.patient_id.values
POS_ABS = (Y == 1) & (DEV == 0)
POS_DEV = (Y == 1) & (DEV == 1)
NEG = Y == 0
NEG_DEV, NEG_NONE = NEG & (DEV == 1), NEG & (DEV == 0)

SCORE, SEEDS, VAL, VALY, THR = {}, {}, {}, {}, {}
for r in RUNS:
    SCORE[r] = L[r]["df"].set_index("image_id").loc[IDS, "score"].values
    THR[r] = L[r]["thr"]
    ss, vv, tt, names = [], [], [], []
    for df, val, thr, name in L[r]["per_seed"]:
        ss.append(df.set_index("image_id").loc[IDS, "score"].values)
        vv.append(val.set_index("image_id").loc[L[r]["val"].image_id, "score"].values)
        tt.append(thr); names.append(name)
    SEEDS[r] = {"test": np.array(ss), "val": np.array(vv), "thr": np.array(tt), "names": names}
    VALY[r] = L[r]["val"].ptx.values


def metrics(s, thr, idx=None):
    """Endpoints on one score vector (optionally restricted to the index multiset idx)."""
    if idx is None:
        idx = np.arange(len(s))
    s, y = s[idx], Y[idx]
    pa, pd_, ng = POS_ABS[idx], POS_DEV[idx], NEG[idx]
    nd, nn = NEG_DEV[idx], NEG_NONE[idx]
    a_abs, a_dev = auc(s[pa], s[ng]), auc(s[pd_], s[ng])
    return {"auroc": auc(s[y == 1], s[ng]), "auc_abs": a_abs, "auc_dev": a_dev, "gap": a_dev - a_abs,
            "devscore": auc(s[nd], s[nn]), "sens_abs": (s[pa] >= thr).mean() if pa.any() else np.nan,
            "spec": (s[ng] < thr).mean()}


KEYS = ["auroc", "auc_abs", "auc_dev", "gap", "devscore", "sens_abs", "spec"]
UPAT, INV = np.unique(PAT, return_inverse=True)
MEMBERS = [np.where(INV == k)[0] for k in range(len(UPAT))]


def boot_indices(rng, keep=None):
    """Patient-cluster resample; keep = boolean image mask defining the analysed subset."""
    mem = MEMBERS if keep is None else [m[keep[m]] for m in MEMBERS if keep[m].any()]
    pick = rng.integers(0, len(mem), len(mem))
    return np.concatenate([mem[k] for k in pick])


# ------------------------------------------------------------------ 1. clustering structure of the NIH test set (R2-7)
sub = POS_ABS | NEG
g = pd.DataFrame({"p": PAT[sub], "y": Y[sub]}).groupby("p").y.agg(["size", "nunique"])
gall = pd.Series(PAT).value_counts()
J["clusters"] = {"n_images": int(len(Y)), "n_patients": int(len(UPAT)),
                 "patients_with_gt1_image": int((gall > 1).sum()), "images_in_multi_image_patients": int(gall[gall > 1].sum()),
                 "abs_auc_subset_images": int(sub.sum()), "abs_auc_subset_patients": int(len(g)),
                 "abs_auc_subset_multi_image_patients": int((g["size"] > 1).sum()),
                 "abs_auc_subset_mixed_label_patients": int((g["nunique"] > 1).sum()),
                 "abs_pos_images": int(POS_ABS.sum()), "abs_pos_patients": int(len(set(PAT[POS_ABS]))),
                 "dev_pos_images": int(POS_DEV.sum()), "dev_pos_patients": int(len(set(PAT[POS_DEV]))),
                 "neg_images": int(NEG.sum()), "neg_patients": int(len(set(PAT[NEG])))}


# ------------------------------------------------------------------ 2. Obuchowski (1997) clustered paired AUROC test (R2-7)
def obuchowski(score_list, pos, neg):
    """Clustered nonparametric AUROC variance/covariance (Obuchowski, Biometrics 1997).
    score_list: list of score vectors (same images); pos/neg: boolean masks. Returns aucs, covariance matrix."""
    sel = pos | neg
    clus = PAT[sel]
    u, inv = np.unique(clus, return_inverse=True)
    I = len(u)
    p_sel, n_sel = pos[sel], neg[sel]
    M, N = p_sel.sum(), n_sel.sum()
    m_i = np.bincount(inv, weights=p_sel.astype(float), minlength=I)
    n_i = np.bincount(inv, weights=n_sel.astype(float), minlength=I)
    th, V10, V01 = [], [], []
    for s in score_list:
        s = s[sel]
        x, y = s[p_sel], s[n_sel]
        psi = (x[:, None] > y[None, :]).astype(float) + 0.5 * (x[:, None] == y[None, :])
        v10 = np.zeros(len(s)); v01 = np.zeros(len(s))
        v10[p_sel] = psi.mean(1)          # per positive, average over all negatives
        v01[n_sel] = psi.mean(0)          # per negative, average over all positives
        t = psi.mean()
        th.append(t)
        V10.append(np.bincount(inv, weights=v10, minlength=I) - m_i * t)
        V01.append(np.bincount(inv, weights=v01, minlength=I) - n_i * t)
    k = len(score_list)
    C = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            s10 = I / ((I - 1) * M) * np.sum(V10[a] * V10[b])
            s01 = I / ((I - 1) * N) * np.sum(V01[a] * V01[b])
            s11 = I / (I - 1) * np.sum(V10[a] * V01[b] + V01[a] * V10[b]) / 2
            C[a, b] = s10 / M + s01 / N + 2 * s11 / (M * N)
    return np.array(th), C


obu = {}
for r in RUNS[1:]:
    th, Cm = obuchowski([SCORE[PROP], SCORE[r]], POS_ABS, NEG)
    v = Cm[0, 0] + Cm[1, 1] - 2 * Cm[0, 1]
    z = (th[0] - th[1]) / np.sqrt(v)
    _, _, d_dl, p_dl = delong_test(Y[POS_ABS | NEG], SCORE[PROP][POS_ABS | NEG], SCORE[r][POS_ABS | NEG])
    obu[r] = {"auc_prop": r4(th[0]), "auc_comp": r4(th[1]), "diff": r4(th[0] - th[1]),
              "se_clustered": r4(np.sqrt(v)), "p_clustered": float(2 * stats.norm.sf(abs(z))),
              "p_delong_independent": float(p_dl)}
for fam, members in [("holm_prespecified_7", PRESPEC_BL), ("holm_all_11", RUNS[1:])]:
    for key in ("p_clustered", "p_delong_independent"):
        for r, p in zip(members, holm([obu[r][key] for r in members])):
            obu[r][f"{key}_{fam}"] = r4(p)
for r in obu:
    obu[r]["p_clustered"] = r4(obu[r]["p_clustered"]); obu[r]["p_delong_independent"] = r4(obu[r]["p_delong_independent"])
J["obuchowski_auc_abs"] = obu


# ------------------------------------------------------------------ 3. full cohort: point estimates with CIs for every model (R2-5)
def boot_all(keep=None, seed=0):
    """Patient-cluster bootstrap of all endpoints for all models with shared resamples."""
    rng = np.random.default_rng(seed)
    out = {r: {k: [] for k in KEYS} for r in RUNS}
    for _ in range(B):
        idx = boot_indices(rng, keep)
        for r in RUNS:
            m = metrics(SCORE[r], THR[r], idx)
            for k in KEYS:
                out[r][k].append(m[k])
    return {r: {k: np.array(v) for k, v in d.items()} for r, d in out.items()}


def summarise(keep=None, seed=0, label=""):
    idx0 = np.arange(len(Y)) if keep is None else np.where(keep)[0]
    bs = boot_all(keep, seed)
    res = {}
    for r in RUNS:
        m = metrics(SCORE[r], THR[r], idx0)
        row = {}
        for k in KEYS:
            v = bs[r][k][~np.isnan(bs[r][k])]
            row[k] = [r4(m[k]), r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))]
        pa = POS_ABS[idx0]
        k_det = int((SCORE[r][idx0][pa] >= THR[r]).sum())
        row["sens_abs_count"] = [k_det, int(pa.sum())]
        row["sens_abs_wilson"] = wilson(k_det, int(pa.sum()))
        ng = NEG[idx0]
        k_sp = int((SCORE[r][idx0][ng] < THR[r]).sum())
        row["spec_count"] = [k_sp, int(ng.sum())]
        row["spec_wilson"] = wilson(k_sp, int(ng.sum()))
        res[r] = row
    # paired differences, proposed minus comparator, same resamples
    pair = {}
    for r in RUNS[1:]:
        row = {}
        for k in ("auc_abs", "gap", "sens_abs", "devscore"):
            pt = metrics(SCORE[PROP], THR[PROP], idx0)[k] - metrics(SCORE[r], THR[r], idx0)[k]
            d = bs[PROP][k] - bs[r][k]
            d = d[~np.isnan(d)]
            row[k] = [r4(pt), r4(np.percentile(d, 2.5)), r4(np.percentile(d, 97.5)), r4(boot_p_two_sided(d))]
        pair[r] = row
    for k in ("auc_abs", "gap", "sens_abs", "devscore"):
        for r, p in zip(RUNS[1:], holm([pair[r][k][3] for r in RUNS[1:]])):
            pair[r][k].append(r4(p))
    n = {"images": int(len(idx0)), "patients": int(len(set(PAT[idx0]))), "pos_dev": int(POS_DEV[idx0].sum()),
         "pos_abs": int(POS_ABS[idx0].sum()), "neg": int(NEG[idx0].sum()), "neg_dev": int(NEG_DEV[idx0].sum()),
         "neg_none": int(NEG_NONE[idx0].sum())}
    return {"n": n, "models": res, "paired_vs_proposed": pair}


J["full_cohort"] = summarise(None, 0)

# ------------------------------------------------------------------ 4. after removal of identified patient overlap (R2-2, R2-5)
pairs = pd.read_csv(INP / "siim_neatx_pairs.csv", dtype={"image_id": str, "nih_id": str})
sp = pd.read_csv("splits/siim.csv", dtype={"image_id": str})
pm = pairs.drop_duplicates(["image_id", "nih_id"]).merge(sp, on="image_id")
pm["nih_patient"] = pm.nih_id.str[:8]
npat = pm.groupby("image_id").nih_patient.nunique()
pm = pm[pm.image_id.isin(npat[npat == 1].index)].drop_duplicates("image_id")
dev_pat = set(pm[pm.split.isin(["train", "val"])].nih_patient)
KEEP = ~pd.Series(IDS).str[:8].isin(dev_pat).values
J["overlap_excluded"] = summarise(KEEP, 0)

# ------------------------------------------------------------------ 5. seed level and hierarchical bootstrap (R2-6, R2-17)
seed_tab, agg = {}, {}
for r in RUNS:
    rows = []
    for j, name in enumerate(SEEDS[r]["names"]):
        m = metrics(SEEDS[r]["test"][j], SEEDS[r]["thr"][j])
        rows.append({"seed": name, **{k: r4(m[k]) for k in KEYS}})
    seed_tab[r] = rows
    df = pd.DataFrame(rows)
    agg[r] = {"n_seeds": len(rows), **{k: {"mean": r4(df[k].mean()), "sd": r4(df[k].std(ddof=1)),
                                           "min": r4(df[k].min()), "max": r4(df[k].max())} for k in KEYS}}
    # the two aggregation conventions for device-absent sensitivity
    agg[r]["sens_abs_seed_averaged_scores"] = r4(metrics(SCORE[r], THR[r])["sens_abs"])
    agg[r]["sens_abs_mean_of_seeds"] = r4(df.sens_abs.mean())
J["per_seed"] = seed_tab
J["per_seed_summary"] = agg


def hier_boot(keep=None, seed=2):
    """Two-stage bootstrap: patients with replacement (shared by all models), then seeds with replacement
    (independently per model). The seed-averaged score and its SIIM-ACR validation threshold are recomputed
    from the resampled seeds in every replicate."""
    rng = np.random.default_rng(seed)
    res = {r: {k: [] for k in ("auc_abs", "gap", "devscore", "sens_abs", "auroc")} for r in RUNS}
    for _ in range(B):
        idx = boot_indices(rng, keep)
        for r in RUNS:
            S = SEEDS[r]
            pick = rng.integers(0, len(S["names"]), len(S["names"]))
            s = S["test"][pick].mean(0)
            thr = threshold_at_specificity(VALY[r], S["val"][pick].mean(0), 0.95)
            m = metrics(s, thr, idx)
            for k in res[r]:
                res[r][k].append(m[k])
    return {r: {k: np.array(v) for k, v in d.items()} for r, d in res.items()}


hb = hier_boot(None, 2)
hier = {}
for r in RUNS:
    m0 = metrics(SCORE[r], THR[r])
    hier[r] = {k: [r4(m0[k]), r4(np.nanpercentile(hb[r][k], 2.5)), r4(np.nanpercentile(hb[r][k], 97.5))] for k in hb[r]}
hpair = {}
for r in RUNS[1:]:
    hpair[r] = {}
    for k in ("auc_abs", "gap", "sens_abs"):
        d = hb[PROP][k] - hb[r][k]; d = d[~np.isnan(d)]
        pt = metrics(SCORE[PROP], THR[PROP])[k] - metrics(SCORE[r], THR[r])[k]
        hpair[r][k] = [r4(pt), r4(np.percentile(d, 2.5)), r4(np.percentile(d, 97.5)), r4(boot_p_two_sided(d))]
for k in ("auc_abs", "gap", "sens_abs"):
    for r, p in zip(RUNS[1:], holm([hpair[r][k][3] for r in RUNS[1:]])):
        hpair[r][k].append(r4(p))
J["hierarchical"] = {"models": hier, "paired_vs_proposed": hpair}

# ------------------------------------------------------------------ 6. checkpoint-selection stability (R2-4)
ck = []
for r in RUNS:
    for j, name in enumerate(SEEDS[r]["names"]):
        log = pd.read_csv(R / r / name / "train_log.csv")
        th = json.load(open(R / r / name / "threshold.json"))
        u = log.val_sens95_untreated.values
        best = u.max()
        sel = int(np.argmax(u))                                  # first epoch reaching the maximum
        near = np.where(u >= best - 1 / 32 - 1e-9)[0]            # within one of 32 validation cases
        auc_sel = int(np.argmax(log.val_auroc.values))
        m_ext = metrics(SEEDS[r]["test"][j], SEEDS[r]["thr"][j])
        ck.append({"run": r, "seed": name, "epochs": int(len(u)), "selected_epoch": sel + 1, "best_val_sens_abs": r4(best),
                   "stored_best": r4(th.get("best", np.nan)),
                   "n_epochs_tied": int((u >= best - 1e-9).sum()), "n_epochs_within_one_case": int(len(near)),
                   "range_epochs_within_one_case": [int(near.min()) + 1, int(near.max()) + 1],
                   "val_auroc_selected": r4(log.val_auroc.values[sel]), "val_auroc_max": r4(log.val_auroc.max()),
                   "epoch_by_val_auroc": auc_sel + 1, "same_epoch_as_auroc_rule": bool(auc_sel == sel),
                   "val_sens_abs_at_auroc_epoch": r4(u[auc_sel]),
                   "nih_sens_abs": r4(m_ext["sens_abs"]), "nih_auc_abs": r4(m_ext["auc_abs"]), "nih_gap": r4(m_ext["gap"])})
ckd = pd.DataFrame(ck)
rho_s = spearmanr(ckd.best_val_sens_abs, ckd.nih_sens_abs)
rho_a = spearmanr(ckd.best_val_sens_abs, ckd.nih_auc_abs)
within = []
for r, g in ckd.groupby("run"):
    if len(g) >= 3:
        within.append({"run": r, "rho_sens": r4(spearmanr(g.best_val_sens_abs, g.nih_sens_abs)[0])})
J["checkpoint_selection"] = {
    "runs": ck,
    "summary": {"n_runs": int(len(ckd)), "selected_epoch_range": [int(ckd.selected_epoch.min()), int(ckd.selected_epoch.max())],
                "median_epochs_within_one_case": float(ckd.n_epochs_within_one_case.median()),
                "range_epochs_within_one_case": [int(ckd.n_epochs_within_one_case.min()), int(ckd.n_epochs_within_one_case.max())],
                "share_runs_with_tie": r4((ckd.n_epochs_tied > 1).mean()),
                "share_same_epoch_as_auroc_rule": r4(ckd.same_epoch_as_auroc_rule.mean()),
                "median_val_auroc_loss_vs_max": r4((ckd.val_auroc_max - ckd.val_auroc_selected).median()),
                "max_val_auroc_loss_vs_max": r4((ckd.val_auroc_max - ckd.val_auroc_selected).max()),
                "best_val_sens_abs_range": [r4(ckd.best_val_sens_abs.min()), r4(ckd.best_val_sens_abs.max())],
                "binomial_se_at_0.75_n32": r4(np.sqrt(0.75 * 0.25 / 32)),
                "spearman_val_vs_nih_sens_abs": [r4(rho_s[0]), r4(rho_s[1])],
                "spearman_val_vs_nih_auc_abs": [r4(rho_a[0]), r4(rho_a[1])],
                "within_config_spearman_sens": within},
    "proposed": ckd[ckd.run == PROP][["seed", "selected_epoch", "best_val_sens_abs", "n_epochs_within_one_case",
                                       "range_epochs_within_one_case", "epoch_by_val_auroc", "nih_sens_abs"]].to_dict("records")}

# ------------------------------------------------------------------ 7. pseudo-label agreement with CIs and representativeness (R2-9)
dr = pd.read_csv(INP / "siim_drain_labels.csv", dtype={"image_id": str})
lab = pd.read_csv("results/device_labels/siim_device_labels.csv", dtype={"image_id": str})
mm = lab.merge(dr[["image_id", "split", "ptx", "Drain"]], on="image_id")
nihpat = pm.set_index("image_id").nih_patient
mm["cluster"] = mm.image_id.map(nihpat).fillna(mm.image_id)


def agree(sub, seed=3):
    d1d1 = int(((sub.Drain == 1) & (sub.device == 1)).sum()); d1d0 = int(((sub.Drain == 1) & (sub.device == 0)).sum())
    d0d1 = int(((sub.Drain == 0) & (sub.device == 1)).sum()); d0d0 = int(((sub.Drain == 0) & (sub.device == 0)).sum())
    est = {"sens_pseudo_for_drain": (d1d1, d1d1 + d1d0), "drain_share_among_pseudo_dev": (d1d1, d1d1 + d0d1),
           "nodrain_but_device": (d0d1, d0d1 + d0d0)}
    out = {"n": int(len(sub)), "n_clusters": int(sub.cluster.nunique())}
    rng = np.random.default_rng(seed)
    cl = sub.cluster.values
    u, inv = np.unique(cl, return_inverse=True)
    mem = [np.where(inv == k)[0] for k in range(len(u))]
    D, V = sub.Drain.values.astype(int), sub.device.values.astype(int)
    bs = {k: [] for k in est}
    for _ in range(B):
        idx = np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])
        d, v = D[idx], V[idx]
        bs["sens_pseudo_for_drain"].append((d & v).sum() / max(1, d.sum()))
        bs["drain_share_among_pseudo_dev"].append((d & v).sum() / max(1, v.sum()))
        bs["nodrain_but_device"].append(((1 - d) & v).sum() / max(1, (1 - d).sum()))
    for k, (a, b) in est.items():
        out[k] = {"k": a, "n": b, "p": r4(a / b), "wilson": wilson(a, b),
                  "cluster_boot": [r4(np.percentile(bs[k], 2.5)), r4(np.percentile(bs[k], 97.5))]}
    return out


J["pseudo_vs_drain_ci"] = {nm: agree(s) for nm, s in [("all", mm), ("ptx_pos", mm[mm.ptx == 1]), ("ptx_neg", mm[mm.ptx == 0])]}
# representativeness: mapped subset versus all other SIIM-ACR images
posset = set(pd.read_csv("results/device_labels/siim_positives_review.csv", dtype={"image_id": str}).image_id)
rep_df = sp.merge(lab[["image_id", "device"]], on="image_id", how="left")
rep_df["ptx"] = rep_df.image_id.isin(posset).astype(int)
rep_df["mapped"] = rep_df.image_id.isin(mm.image_id)


def rep(d):
    return {"n": int(len(d)), "ptx_rate": r4(d.ptx.mean()), "device_rate": r4(d.device.mean()),
            "device_rate_pos": r4(d[d.ptx == 1].device.mean()), "device_rate_neg": r4(d[d.ptx == 0].device.mean()),
            "split_share": {k: r4(v) for k, v in d.split.value_counts(normalize=True).items()}}


J["pseudo_mapped_representativeness"] = {"mapped": rep(rep_df[rep_df.mapped]), "not_mapped": rep(rep_df[~rep_df.mapped]),
                                         "all": rep(rep_df)}
# drain share among pseudo-labelled device-bearing images, re-weighted to the full development set:
# P(stratum | device) from all 10,675 SIIM-ACR images, stratum-specific shares from the mapped subset
st = {}
prev = rep_df.ptx.mean()
dpos, dneg = rep_df[rep_df.ptx == 1].device.mean(), rep_df[rep_df.ptx == 0].device.mean()
w_pos = prev * dpos / (prev * dpos + (1 - prev) * dneg)
pp = J["pseudo_vs_drain_ci"]["ptx_pos"]["drain_share_among_pseudo_dev"]["p"]
pn = J["pseudo_vs_drain_ci"]["ptx_neg"]["drain_share_among_pseudo_dev"]["p"]
st["drain_share_among_pseudo_dev"] = r4(w_pos * pp + (1 - w_pos) * pn)
st["weight_ptx_pos_among_device"] = r4(w_pos)
J["pseudo_vs_drain_standardised_to_siim_prevalence"] = st

# ------------------------------------------------------------------ 8. acquisition-related metadata on NIH (R2-11)
md = pd.read_csv("results/r1/nih_metadata.csv", dtype={"image_id": str}).set_index("image_id").loc[IDS]
spc = md["OriginalImagePixelSpacing[x"].round(3).values
w = md["OriginalImage[Width"].values
acq = {"pixel_spacing_values": pd.Series(spc).value_counts().to_dict()}
tab = pd.DataFrame({"spacing": spc, "view": md["View Position"].values, "neg_dev": NEG_DEV, "neg_none": NEG_NONE,
                    "pos_dev": POS_DEV, "pos_abs": POS_ABS})
acq["by_spacing"] = tab.groupby("spacing")[["neg_dev", "neg_none", "pos_dev", "pos_abs"]].sum().astype(int).to_dict("index")
acq["by_spacing_view"] = tab.groupby(["spacing", "view"])[["neg_dev", "neg_none", "pos_dev", "pos_abs"]].sum().astype(int) \
    .reset_index().to_dict("records")
main_sp = pd.Series(spc).value_counts().index[0]
within = {}
msk = spc == main_sp
for r in RUNS:
    s = SCORE[r]
    within[r] = {"gap": r4(auc(s[POS_DEV & msk], s[NEG & msk]) - auc(s[POS_ABS & msk], s[NEG & msk])),
                 "devscore": r4(auc(s[NEG_DEV & msk], s[NEG_NONE & msk]))}
acq["within_main_spacing"] = {"spacing": float(main_sp), "n_pos_dev": int((POS_DEV & msk).sum()),
                              "n_pos_abs": int((POS_ABS & msk).sum()), "n_neg_dev": int((NEG_DEV & msk).sum()),
                              "n_neg_none": int((NEG_NONE & msk).sum()), "models": within}
# adjusted logit model among positives: device + view + sex + age + detector (pixel-spacing category)
import statsmodels.formula.api as smf  # noqa: E402
adj = {}
base = pd.DataFrame({"dev": DEV, "y": Y, "pat": PAT, "view": md["View Position"].values, "sex": md["Patient Gender"].values,
                     "age": pd.to_numeric(md["Patient Age"], errors="coerce").values, "spc": pd.Series(spc).astype(str).values})
for r in RUNS:
    d = base.copy()
    d["z"] = np.log(np.clip(SCORE[r], 1e-6, 1 - 1e-6) / (1 - np.clip(SCORE[r], 1e-6, 1 - 1e-6)))
    d = d[(d.y == 1) & (d.dev >= 0)].dropna()
    f0 = smf.ols("z ~ dev + C(view) + C(sex) + age", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.pat.astype("category").cat.codes})
    f1 = smf.ols("z ~ dev + C(view) + C(sex) + age + C(spc)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.pat.astype("category").cat.codes})
    adj[r] = {"coef_view_sex_age": r4(f0.params["dev"]),
              "coef_plus_spacing": r4(f1.params["dev"]), "ci_plus_spacing": [r4(f1.conf_int().loc["dev", 0]), r4(f1.conf_int().loc["dev", 1])],
              "p_plus_spacing": r4(f1.pvalues["dev"])}
acq["logit_adjusted"] = adj
J["acquisition"] = acq

json.dump(J, open(OUT / "cpu.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("written", OUT / "cpu.json")
