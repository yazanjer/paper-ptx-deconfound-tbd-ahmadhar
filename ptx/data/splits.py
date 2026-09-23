"""Near-duplicate detection and frozen, patient-level splits.

    python -m ptx.data.splits --data /workspace/data --dev candid --out splits/

Writes splits/<dataset>.csv (image_id, split) and splits/duplicates.csv.
Splits are generated ONCE, committed to git, and never regenerated
(re-running refuses to overwrite unless --force).

Split roles
    dev dataset      : train / val / test_internal   (70/10/20 by patient, stratified by ptx x treated)
    nih              : its own released patient-wise split -> pilot_train / pilot_val / test
    other externals  : test
Images that are near-duplicates of any dev image are dropped from external test
sets (split = "excluded_dup"); duplicates inside the dev set are forced into the
same patient group.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def phash_all(data: Path, datasets):
    import imagehash
    rows = []
    for ds in datasets:
        man = pd.read_csv(data / ds / "manifest.csv", dtype={"image_id": str})
        for iid in man.image_id:
            h = imagehash.phash(Image.open(data / ds / "img" / f"{iid}.png").resize((256, 256)))
            rows.append((ds, str(iid), int(str(h), 16)))
    return pd.DataFrame(rows, columns=["dataset", "image_id", "phash"])


def find_duplicates(h: pd.DataFrame, max_dist: int = 4) -> pd.DataFrame:
    """All pairs with Hamming distance <= max_dist (vectorised popcount, O(n^2/64))."""
    v = h.phash.values.astype(np.uint64)
    pairs = []
    for i in range(len(v)):
        x = np.bitwise_xor(v[i + 1:], v[i])
        d = np.array([bin(int(t)).count("1") for t in x]) if len(x) < 2000 else np.unpackbits(
            x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.where(d <= max_dist)[0]:
            k = i + 1 + j
            pairs.append((h.dataset.iat[i], h.image_id.iat[i], h.dataset.iat[k], h.image_id.iat[k], int(d[j])))
    return pd.DataFrame(pairs, columns=["ds_a", "id_a", "ds_b", "id_b", "hamming"])


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def dev_split(man: pd.DataFrame, dups: pd.DataFrame, seed: int = 2026, frac=(0.7, 0.1, 0.2)) -> pd.Series:
    from sklearn.model_selection import StratifiedGroupKFold
    uf = UF()
    for p in man.patient_id.astype(str):
        uf.f(p)
    pid = dict(zip(man.image_id.astype(str), man.patient_id.astype(str)))
    for r in dups.itertuples():
        if r.id_a in pid and r.id_b in pid and r.ds_a == r.ds_b == man.dataset.iat[0]:
            uf.u(pid[r.id_a], pid[r.id_b])
    groups = man.patient_id.astype(str).map(uf.f).values
    strata = (man.ptx.astype(int) * 3 + man.treated.fillna(-1).astype(int) + 1).values
    # 10 folds -> 7 train, 1 val, 2 test
    sgk = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.zeros(len(man), dtype=int)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(man)), strata, groups)):
        fold[te] = k
    n_tr, n_va = round(frac[0] * 10), round(frac[1] * 10)
    split = np.where(fold < n_tr, "train", np.where(fold < n_tr + n_va, "val", "test_internal"))
    return pd.Series(split, index=man.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dev", default="candid")
    ap.add_argument("--externals", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    present = [d for d in ["candid", "siim", "nih", "ptx498"] if (a.data / d / "manifest.csv").exists()]
    externals = a.externals if a.externals is not None else [d for d in present if d != a.dev]
    if a.dev == "none":  # externals only (e.g. NIH pilot before the dev set is available)
        for ds in externals:
            m = pd.read_csv(a.data / ds / "manifest.csv", dtype={"image_id": str})
            m["split"] = m.source_split.map({"train": "pilot_train", "validation": "pilot_val", "test": "test"}) if ds == "nih" else "test"
            m[["image_id", "split"]].to_csv(a.out / f"{ds}.csv", index=False)
            print(ds, m.split.value_counts().to_dict())
        return
    if (a.out / f"{a.dev}.csv").exists() and not a.force:
        raise SystemExit(f"{a.out}/{a.dev}.csv exists: splits are frozen. Use --force only before any training.")
    h = phash_all(a.data, [a.dev] + externals)
    dups = find_duplicates(h, a.max_dist)
    dups.to_csv(a.out / "duplicates.csv", index=False)
    print(f"{len(dups)} near-duplicate pairs (hamming <= {a.max_dist})")

    man = pd.read_csv(a.data / a.dev / "manifest.csv", dtype={"image_id": str})
    man["split"] = dev_split(man, dups)
    man[["image_id", "split"]].to_csv(a.out / f"{a.dev}.csv", index=False)
    print(a.dev, man.groupby(["split", "ptx"]).size().to_dict())

    dev_ids = set(man.image_id)
    cross = set()
    for r in dups.itertuples():
        if r.ds_a == a.dev and r.id_a in dev_ids and r.ds_b != a.dev:
            cross.add((r.ds_b, r.id_b))
        if r.ds_b == a.dev and r.id_b in dev_ids and r.ds_a != a.dev:
            cross.add((r.ds_a, r.id_a))
    for ds in externals:
        m = pd.read_csv(a.data / ds / "manifest.csv", dtype={"image_id": str})
        if ds == "nih":
            m["split"] = m.source_split.map({"train": "pilot_train", "validation": "pilot_val", "test": "test"})
        else:
            m["split"] = "test"
        m.loc[[(ds, i) in cross for i in m.image_id], "split"] = "excluded_dup"
        m[["image_id", "split"]].to_csv(a.out / f"{ds}.csv", index=False)
        print(ds, m.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
