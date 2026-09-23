"""Counts per dataset x ptx x treated (goes into the paper's dataset table)."""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path("results/data_report.md"))
a = ap.parse_args()
L = ["# Data report", ""]
for ds in ["candid", "siim", "nih", "ptx498"]:
    f = a.data / ds / "manifest.csv"
    if not f.exists():
        L.append(f"- {ds}: not prepared")
        continue
    m = pd.read_csv(f)
    L += [f"## {ds}", "", f"{len(m)} images, {int(m.ptx.sum())} pneumothorax, {m.patient_id.nunique()} patients", "",
          pd.crosstab(m.ptx, m.treated.fillna(-1)).to_markdown(), ""]
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("\n".join(L))
print("\n".join(L))
