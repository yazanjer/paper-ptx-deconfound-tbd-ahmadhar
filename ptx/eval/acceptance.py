"""Pre-registered acceptance checker: A1-A8, sanity checks, SOTA table, ablation table.

    python -m ptx.eval.acceptance --spec configs/acceptance.yaml --runs results/runs --out results

Reads every results/runs/<run>/seed*/preds_*.csv, averages scores over seeds per image,
applies the dev-validation threshold (95% specificity) and writes:
    results/acceptance_report.md, results/acceptance.json,
    results/tables/sota.csv, results/tables/ablation.csv, results/tables/per_seed.csv
Nothing here tunes anything: thresholds come from dev validation only.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ptx.eval.metrics import auroc, ece, fit_temperature, sens_spec_at, threshold_at_specificity
from ptx.eval.size import bts_class, is_small, tertile_class
from ptx.eval.stats import boot_p_two_sided, cluster_bootstrap, delong_test, holm, noninferiority

N_BOOT = 2000


# --------------------------------------------------------------------------- loading
def seeds_of(runs: Path, run: str):
    return sorted(glob.glob(str(runs / run / "seed*" / "DONE")))


def load_run(runs: Path, run: str, dev: str, ds: str, split: str):
    """Returns (seed-averaged df with columns score/logit, threshold, per-seed list of (df, thr))."""
    per = []
    for done in seeds_of(runs, run):
        d = Path(done).parent
        f = d / f"preds_{ds}_{split}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype={"image_id": str})
        val = pd.read_csv(d / f"preds_{dev}_val.csv", dtype={"image_id": str})
        thr = json.load(open(d / "threshold.json"))["threshold_spec95"]
        per.append((df, val, thr, d.name))
    if not per:
        return None
    keys = ["image_id"]
    avg = per[0][0].drop(columns=["score", "logit"]).copy()
    avg["score"] = np.mean([p[0].set_index("image_id").loc[avg.image_id, "score"].values for p in per], axis=0)
    avg["logit"] = np.mean([p[0].set_index("image_id").loc[avg.image_id, "logit"].values for p in per], axis=0)
    vavg = per[0][1][keys + ["ptx"]].copy()
    vavg["score"] = np.mean([p[1].set_index("image_id").loc[vavg.image_id, "score"].values for p in per], axis=0)
    vavg["logit"] = np.mean([p[1].set_index("image_id").loc[vavg.image_id, "logit"].values for p in per], axis=0)
    thr = threshold_at_specificity(vavg.ptx.values, vavg.score.values, 0.95)
    return {"df": avg, "val": vavg, "thr": thr, "per_seed": per}


# --------------------------------------------------------------------------- metrics on one prediction table
def masks(df):
    pos, neg = df.ptx.values == 1, df.ptx.values == 0
    unt = pos & (df.treated.values == 0)
    trt = pos & (df.treated.values == 1)
    return pos, neg, unt, trt


def sens_untreated(df, thr, idx=None):
    d = df if idx is None else df.iloc[idx]
    _, _, unt, _ = masks(d)
    return float((d.score.values[unt] >= thr).mean()) if unt.any() else np.nan


def auc_sub(df, which, idx=None):
    d = df if idx is None else df.iloc[idx]
    pos, neg, unt, trt = masks(d)
    sel = {"untreated": unt, "treated": trt, "all": pos}[which]
    m = sel | neg
    return auroc(d.ptx.values[m], d.score.values[m])


def auc_gap(df, idx=None):
    return auc_sub(df, "treated", idx) - auc_sub(df, "untreated", idx)


def sens_gap(df, thr, idx=None):
    d = df if idx is None else df.iloc[idx]
    _, _, unt, trt = masks(d)
    if not unt.any() or not trt.any():
        return np.nan
    return float((d.score.values[trt] >= thr).mean() - (d.score.values[unt] >= thr).mean())


def summarize(df, thr):
    pos, neg, unt, trt = masks(df)
    se, sp = sens_spec_at(df.ptx.values, df.score.values, thr)
    out = {"n": len(df), "n_pos": int(pos.sum()), "n_untreated_pos": int(unt.sum()), "n_treated_pos": int(trt.sum()),
           "auroc": auroc(df.ptx.values, df.score.values) if neg.any() else np.nan,
           "sens": se, "spec": sp, "sens_untreated": sens_untreated(df, thr),
           "auroc_untreated": auc_sub(df, "untreated") if neg.any() else np.nan,
           "auroc_treated": auc_sub(df, "treated") if neg.any() else np.nan,
           "sens_gap": sens_gap(df, thr)}
    out["auc_gap"] = out["auroc_treated"] - out["auroc_untreated"]
    if "dice" in df and pos.any() and df.loc[pos, "dice"].notna().any():
        out["dice_pos"] = float(df.loc[pos, "dice"].mean())
    return out


# --------------------------------------------------------------------------- checker
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/acceptance.yaml")
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    S = yaml.safe_load(open(a.spec))
    dev = S["dev"]
    prop = S["proposed"]
    bases = S["baselines"]
    externals = S["primary_externals"]  # list of {dataset, split}
    (a.out / "tables").mkdir(parents=True, exist_ok=True)

    R = {}  # (run, ds) -> loaded
    all_sets = externals + S.get("secondary_sets", [])
    for run in [prop] + bases + [x["run"] for x in S.get("ablations", [])]:
        for e in all_sets:
            r = load_run(a.runs, run, dev, e["dataset"], e["split"])
            if r is not None:
                R[(run, e["dataset"], e["split"])] = r

    # ---- SOTA / summary table
    rows = []
    for (run, ds, sp), r in R.items():
        s = summarize(r["df"], r["thr"])
        s.update(run=run, dataset=ds, split=sp, seeds=len(r["per_seed"]))
        rows.append(s)
    table = pd.DataFrame(rows)
    table.to_csv(a.out / "tables" / "summary_seed_averaged.csv", index=False)

    # ---- per-seed table
    prow = []
    for (run, ds, sp), r in R.items():
        for df, val, thr, seed in r["per_seed"]:
            s = summarize(df, thr)
            s.update(run=run, dataset=ds, split=sp, seed=seed)
            prow.append(s)
    per_seed = pd.DataFrame(prow)
    per_seed.to_csv(a.out / "tables" / "per_seed.csv", index=False)

    res = {"criteria": {}, "sanity": {}, "notes": []}

    def have(run, e):
        return (run, e["dataset"], e["split"]) in R

    # ---- A1, A2, A3, A7, A8 per primary external
    a1, a2, a3p, a3d, a7, a8 = [], [], [], [], [], []
    for e in externals:
        key = (prop, e["dataset"], e["split"])
        if key not in R:
            res["notes"].append(f"missing proposed predictions for {e}")
            continue
        P = R[key]
        dfp, tp = P["df"], P["thr"]
        clusters = dfp.patient_id.astype(str).values
        avail = [b for b in bases if have(b, e)]
        if not avail:
            res["notes"].append(f"no baselines for {e}")
            continue
        # align all tables on image order of proposed
        B = {b: R[(b, e["dataset"], e["split"])] for b in avail}
        for b in avail:
            B[b]["df"] = B[b]["df"].set_index("image_id").loc[dfp.image_id].reset_index()
        # A1: best baseline by untreated sensitivity
        sb = {b: sens_untreated(B[b]["df"], B[b]["thr"]) for b in avail}
        best = max(sb, key=lambda k: -1 if sb[k] != sb[k] else sb[k])
        diff_fn = lambda idx: sens_untreated(dfp, tp, idx) - sens_untreated(B[best]["df"], B[best]["thr"], idx)
        pt, lo, hi, _ = cluster_bootstrap(diff_fn, clusters, N_BOOT)
        a1.append({"set": e, "proposed": sens_untreated(dfp, tp), "best_baseline": best, "baseline": sb[best],
                   "diff": pt, "ci": [lo, hi], "pass": bool(pt >= 0.05 and lo > 0)})
        # A2: smallest-gap baseline
        gb = {b: auc_gap(B[b]["df"]) for b in avail}
        bg = min(gb, key=lambda k: np.inf if gb[k] != gb[k] else gb[k])
        gp = auc_gap(dfp)
        red = 1 - gp / gb[bg] if gb[bg] and gb[bg] > 0 else np.nan
        a2.append({"set": e, "gap_proposed": gp, "gap_baseline": gb[bg], "baseline": bg, "reduction": red,
                   "sens_gap_proposed": sens_gap(dfp, tp), "pass": bool(red == red and red >= 0.5)})
        # A3: DeLong untreated AUROC vs every baseline, Holm-adjusted
        pos, neg, unt, _ = masks(dfp)
        m = unt | neg
        tests = {b: delong_test(dfp.ptx.values[m], dfp.score.values[m], B[b]["df"].score.values[m]) for b in avail}
        padj = dict(zip(avail, holm([tests[b][3] for b in avail])))
        ab = max(avail, key=lambda b: tests[b][1])
        a3d.append({"set": e, "auc_proposed": tests[ab][0], "best_baseline": ab, "auc_baseline": tests[ab][1],
                    "diff": tests[ab][2], "p_holm": padj[ab],
                    "pass": bool(tests[ab][2] >= 0.02 and padj[ab] < 0.05)})
        # A7: ECE after temperature scaling (T from dev validation, seed-averaged logits)
        T = fit_temperature(P["val"].ptx.values, P["val"].logit.values)
        pcal = 1 / (1 + np.exp(-dfp.logit.values / T))
        a7.append({"set": e, "T": T, "ece": ece(dfp.ptx.values, pcal), "pass": bool(ece(dfp.ptx.values, pcal) <= 0.05)})
        # A8: seed stability of A1 metric
        pv = [sens_untreated(df, thr) for df, _, thr, _ in P["per_seed"]]
        bv = np.mean([sens_untreated(df, thr) for df, _, thr, _ in B[best]["per_seed"]])
        wins = int(sum(v > bv for v in pv))
        a8.append({"set": e, "per_seed": pv, "sd": float(np.std(pv, ddof=1)) if len(pv) > 1 else np.nan,
                   "wins_vs_best_baseline_mean": wins, "n_seeds": len(pv),
                   "pass": bool(len(pv) >= 5 and np.std(pv, ddof=1) <= 0.02 and wins >= 4)})

    min_prim = S.get("min_primary_sets", 2)  # Plan B (Amendment 1): one primary external set
    res["criteria"]["A1"] = {"per_set": a1, "pass": bool(a1) and len(a1) >= min_prim and all(x["pass"] for x in a1)}
    res["criteria"]["A2"] = {"per_set": a2, "pass": bool(a2) and len(a2) >= min_prim and all(x["pass"] for x in a2)}
    res["criteria"]["A3"] = {"per_set": a3d, "pass": any(x["pass"] for x in a3d)}
    res["criteria"]["A7"] = {"per_set": a7, "pass": bool(a7) and all(x["pass"] for x in a7)}
    res["criteria"]["A8"] = {"per_set": a8, "pass": bool(a8) and all(x["pass"] for x in a8)}

    # ---- A4 small pneumothorax (sets with reference masks and negatives)
    a4 = []
    for e in all_sets:
        key = (prop, e["dataset"], e["split"])
        if key not in R or "gt_area_frac" not in R[key]["df"]:
            continue
        dfp, tp = R[key]["df"], R[key]["thr"]
        pos = dfp.ptx.values == 1
        small = np.zeros(len(dfp), bool)
        small[pos] = is_small(dfp[pos])
        avail = [b for b in bases if have(b, e)]
        if not avail or not small.any():
            continue
        sp = float((dfp.score.values[small] >= tp).mean())
        sbs = {}
        for b in avail:
            bd = R[(b, e["dataset"], e["split"])]
            bdf = bd["df"].set_index("image_id").loc[dfp.image_id].reset_index()
            sbs[b] = float((bdf.score.values[small] >= bd["thr"]).mean())
        best = max(sbs, key=sbs.get)
        a4.append({"set": e, "n_small": int(small.sum()), "proposed": sp, "best_baseline": best,
                   "baseline": sbs[best], "diff": sp - sbs[best], "win": sp - sbs[best] >= 0.03,
                   "pass": bool(sp >= sbs[best])})
    res["criteria"]["A4"] = {"per_set": a4, "pass": any(x["pass"] for x in a4)}

    # ---- A5 Dice non-inferiority and A6 size agreement
    a5, a6 = [], []
    seg_bases = S.get("segmentation_baselines", bases)
    for e in S.get("dice_sets", []):
        key = (prop, e["dataset"], e["split"])
        if key not in R or "dice" not in R[key]["df"]:
            continue
        dfp = R[key]["df"]
        pos = dfp.ptx.values == 1
        cl = dfp.patient_id.astype(str).values[pos]
        best, bd = None, -1
        for b in seg_bases:
            if have(b, e) and "dice" in R[(b, e["dataset"], e["split"])]["df"]:
                bdf = R[(b, e["dataset"], e["split"])]["df"].set_index("image_id").loc[dfp.image_id].reset_index()
                v = bdf.dice.values[pos].mean()
                if v > bd:
                    best, bd, bvals = b, v, bdf.dice.values[pos]
        if best is None:
            continue
        pvals = dfp.dice.values[pos]
        _, lo, hi, samp = cluster_bootstrap(lambda idx: pvals[idx].mean() - bvals[idx].mean(), cl, N_BOOT)
        ok, lb = noninferiority(samp, -0.01)
        a5.append({"set": e, "dice_proposed": float(pvals.mean()), "best_baseline": best, "dice_baseline": float(bd),
                   "lower_bound_diff": lb, "pass": ok})
        # A6: size class agreement among true positives with a predicted lesion
        tpm = pos & (dfp.pred_area_frac.values > 0)
        d = dfp[tpm]
        if len(d) >= 10:
            if d.gt_rim_mm.notna().all():
                g, p = bts_class(d.gt_rim_mm.values), bts_class(d.pred_rim_mm.values)
                rule = "BTS rim >= 20 mm"
            else:
                cuts = np.quantile(d.gt_area_frac, [1 / 3, 2 / 3])
                g, p = tertile_class(d.gt_area_frac, cuts), tertile_class(d.pred_area_frac, cuts)
                rule = "area-fraction tertiles"
            k = cohen_kappa_score(g, p, weights="linear")
            a6.append({"set": e, "n": int(len(d)), "rule": rule, "kappa": float(k), "pass": bool(k >= 0.70)})
    res["criteria"]["A5"] = {"per_set": a5, "pass": bool(a5) and all(x["pass"] for x in a5)}
    res["criteria"]["A6"] = {"per_set": a6, "pass": len(a6) >= 2 and all(x["pass"] for x in a6)}

    # ---- sanity checks
    occ = []
    for e in all_sets:
        f = sorted(glob.glob(str(a.runs / prop / "seed*" / f"preds_{e['dataset']}_{e['split']}_tubeonly.csv")))
        if not f:
            continue
        def tube_only_rate(run):
            rates = []
            for ff in sorted(glob.glob(str(a.runs / run / "seed*" / f"preds_{e['dataset']}_{e['split']}_tubeonly.csv"))):
                d = pd.read_csv(ff)
                thr = json.load(open(Path(ff).parent / "threshold.json"))["threshold_spec95"]
                rates.append({"pos_rate": float((d.score >= thr).mean()), "auc_ptx": auroc(d.ptx, d.score)})
            return pd.DataFrame(rates).mean().to_dict() if rates else None
        pr = tube_only_rate(prop)
        br = {b: tube_only_rate(b) for b in bases}
        br = {k: v for k, v in br.items() if v}
        if br:
            worst_base = min(v["pos_rate"] for v in br.values())
            occ.append({"set": e, "proposed": pr, "baselines": br,
                        "pass": bool(pr["pos_rate"] < worst_base and (pr["auc_ptx"] != pr["auc_ptx"] or pr["auc_ptx"] <= 0.60))})
    res["sanity"]["occlusion"] = {"per_set": occ, "pass": bool(occ) and all(x["pass"] for x in occ)}

    probe = []
    for e in all_sets:
        def probe_auc(run):
            aucs = []
            for ff in sorted(glob.glob(str(a.runs / run / "seed*" / f"feats_{e['dataset']}_{e['split']}.npz"))):
                z = np.load(ff, allow_pickle=True)
                pr = pd.read_csv(Path(ff).parent / f"preds_{e['dataset']}_{e['split']}.csv", dtype={"image_id": str})
                y = pr.treated.values
                k = y >= 0
                if k.sum() < 50 or len(np.unique(y[k])) < 2:
                    continue
                X = z["feats"].astype(np.float32)[k]
                p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), X, y[k],
                                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
                aucs.append(auroc(y[k], p))
            return float(np.mean(aucs)) if aucs else None
        pp = probe_auc(prop)
        if pp is None:
            continue
        bb = {b: probe_auc(b) for b in bases}
        bb = {k: v for k, v in bb.items() if v is not None}
        if bb:
            probe.append({"set": e, "proposed": pp, "baselines": bb, "pass": bool(pp < min(bb.values()))})
    res["sanity"]["device_probe"] = {"per_set": probe, "pass": bool(probe) and all(x["pass"] for x in probe)}

    leak = json.load(open(a.out / "leakage_check.json")) if (a.out / "leakage_check.json").exists() else {"pass": None}
    res["sanity"]["leakage"] = leak

    # ---- ablations
    abl_rows = []
    full = S.get("ablation_reference", prop)
    for ab in S.get("ablations", []):
        for e in externals:
            k_full, k_ab = (full, e["dataset"], e["split"]), (ab["run"], e["dataset"], e["split"])
            if k_full not in R or k_ab not in R:
                continue
            F_, A_ = R[k_full], R[k_ab]
            fv = {s: sens_untreated(df, thr) for df, _, thr, s in F_["per_seed"]}
            av = {s: sens_untreated(df, thr) for df, _, thr, s in A_["per_seed"]}
            common = sorted(set(fv) & set(av))
            drops = [fv[s] - av[s] for s in common]
            adf = A_["df"].set_index("image_id").loc[F_["df"].image_id].reset_index()
            diff_fn = lambda idx: sens_untreated(F_["df"], F_["thr"], idx) - sens_untreated(adf, A_["thr"], idx)
            pt, lo, hi, samp = cluster_bootstrap(diff_fn, F_["df"].patient_id.astype(str).values, N_BOOT)
            abl_rows.append({"ablation": ab["run"], "removed": ab.get("removed", ""), "dataset": e["dataset"],
                             "full_mean": float(np.mean(list(fv.values()))), "ablated_mean": float(np.mean(list(av.values()))),
                             "mean_drop": float(np.mean(drops)) if drops else np.nan,
                             "seeds_with_drop": int(sum(d > 0 for d in drops)), "n_paired_seeds": len(drops),
                             "diff_seed_avg": pt, "ci_lo": lo, "ci_hi": hi, "p": boot_p_two_sided(samp)})
    abl = pd.DataFrame(abl_rows)
    if len(abl):
        abl["p_holm"] = np.nan
        for ds in abl.dataset.unique():
            m = abl.dataset == ds
            abl.loc[m, "p_holm"] = holm(abl.loc[m, "p"].values)
        verdict = []
        for run in abl.ablation.unique():
            r = abl[abl.ablation == run]
            ok = bool(len(r) >= S.get("min_ablation_sets", 2) and (r.mean_drop > 0).all() and (r.seeds_with_drop >= 4).all())
            if run == S.get("base_ablation"):
                ok = ok and bool((r.ci_lo > 0).all())
            verdict.append({"ablation": run, "positive": ok})
        res["ablation_verdicts"] = verdict
    abl.to_csv(a.out / "tables" / "ablation.csv", index=False)

    crit = res["criteria"]
    primary = crit["A1"]["pass"] and crit["A2"]["pass"]
    n_sec = sum(bool(crit[k]["pass"]) for k in ["A3", "A4", "A5", "A6", "A7", "A8"])
    sanity_ok = all(res["sanity"][k].get("pass") for k in S.get("sanity_checks", ["occlusion", "device_probe", "leakage"]))
    res["overall"] = {"primary_pass": primary, "secondary_passed": n_sec, "sanity_pass": sanity_ok,
                      "ACCEPTED_AS_POSITIVE": bool(primary and n_sec >= 4 and sanity_ok)}
    json.dump(res, open(a.out / "acceptance.json", "w"), indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    write_report(res, table, abl, a.out / "acceptance_report.md")
    print(json.dumps(res["overall"], indent=2))


def write_report(res, table, abl, path):
    L = ["# Acceptance report", "", "Generated by `ptx.eval.acceptance` against the frozen PREREGISTRATION.md.", ""]
    o = res["overall"]
    L += [f"**Accepted as positive: {o['ACCEPTED_AS_POSITIVE']}** (primary {o['primary_pass']}, "
          f"secondary {o['secondary_passed']}/6, sanity {o['sanity_pass']})", "", "| Criterion | Pass |", "| --- | --- |"]
    for k, v in res["criteria"].items():
        L.append(f"| {k} | {v['pass']} |")
    for k, v in res["sanity"].items():
        L.append(f"| sanity: {k} | {v.get('pass')} |")
    L += ["", "## Seed-averaged results", "", table.round(4).to_markdown(index=False) if len(table) else "(none)", ""]
    L += ["## Ablations", "", abl.round(4).to_markdown(index=False) if len(abl) else "(none)", ""]
    if res.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in res["notes"]]
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
