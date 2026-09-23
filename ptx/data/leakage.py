"""Leakage check (sanity criterion): zero patient overlap across dev splits and no dev/external
near-duplicates left in any test split. Writes results/leakage_check.json.

    python -m ptx.data.leakage --data /workspace/data --splits splits --dev candid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--dev", default="candid")
    ap.add_argument("--out", type=Path, default=Path("results/leakage_check.json"))
    a = ap.parse_args()
    man = pd.read_csv(a.data / a.dev / "manifest.csv", dtype={"image_id": str}).merge(
        pd.read_csv(a.splits / f"{a.dev}.csv", dtype={"image_id": str}), on="image_id")
    pats = {s: set(man.loc[man.split == s, "patient_id"].astype(str)) for s in man.split.unique()}
    overlap = {f"{x}&{y}": len(pats[x] & pats[y]) for i, x in enumerate(sorted(pats)) for y in sorted(pats)[i + 1:]}
    dups = pd.read_csv(a.splits / "duplicates.csv", dtype={"id_a": str, "id_b": str})
    split_of = {}
    for f in a.splits.glob("*.csv"):
        if f.name == "duplicates.csv":
            continue
        for r in pd.read_csv(f, dtype={"image_id": str}).itertuples():
            split_of[(f.stem, r.image_id)] = r.split
    bad = 0
    for r in dups.itertuples():
        sa, sb = split_of.get((r.ds_a, r.id_a)), split_of.get((r.ds_b, r.id_b))
        if sa is None or sb is None or "excluded" in (sa + sb):
            continue
        train_like = {"train", "val", "pilot_train", "pilot_val"}
        if (sa in train_like and sb not in train_like) or (sb in train_like and sa not in train_like):
            bad += 1
    res = {"patient_overlap": overlap, "train_test_near_duplicates": bad,
           "pass": all(v == 0 for v in overlap.values()) and bad == 0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
