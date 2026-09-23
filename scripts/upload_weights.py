"""Upload model weights (best.pt) as GitHub release assets, and optionally to a Zenodo deposition.

GitHub: needs GITHUB_TOKEN (Contents: read/write). Creates/uses release tag `weights-v1`.
Zenodo (optional): needs ZENODO_TOKEN; creates a *draft* deposition you publish yourself.
    python scripts/upload_weights.py --runs results/runs [--only full bl_unet_effb4] [--zenodo]
"""
import argparse
import glob
import os
from pathlib import Path

import requests

REPO = "yazanjer/ptx-deconfound"
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path("results/runs"))
ap.add_argument("--only", nargs="*")
ap.add_argument("--tag", default="weights-v1")
ap.add_argument("--zenodo", action="store_true")
a = ap.parse_args()
files = [Path(p) for p in sorted(glob.glob(str(a.runs / "*" / "seed*" / "best.pt")))
         if not a.only or Path(p).parts[-3] in a.only]
files = [f for f in files if not f.parent.parent.name.startswith("pilot_")]
print(f"{len(files)} weight files")
tok = os.environ.get("GITHUB_TOKEN")
if tok and files:
    H = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}", headers=H)
    if r.status_code == 404:
        r = requests.post(f"https://api.github.com/repos/{REPO}/releases", headers=H,
                          json={"tag_name": a.tag, "name": "Model weights", "draft": False, "prerelease": True,
                                "body": "best.pt per run and seed (asset name = <run>__<seed>.pt)."})
    rel = r.json()
    have = {x["name"] for x in rel.get("assets", [])}
    up = rel["upload_url"].split("{")[0]
    for f in files:
        name = f"{f.parent.parent.name}__{f.parent.name}.pt"
        if name in have:
            continue
        with open(f, "rb") as fh:
            rr = requests.post(up, params={"name": name}, data=fh,
                               headers={**H, "Content-Type": "application/octet-stream"})
        print(name, rr.status_code)
ztok = os.environ.get("ZENODO_TOKEN")
if a.zenodo and ztok and files:
    Z = "https://zenodo.org/api/deposit/depositions"
    d = requests.post(Z, params={"access_token": ztok}, json={}).json()
    bucket = d["links"]["bucket"]
    for f in files:
        name = f"{f.parent.parent.name}__{f.parent.name}.pt"
        with open(f, "rb") as fh:
            requests.put(f"{bucket}/{name}", data=fh, params={"access_token": ztok})
    print("Zenodo draft deposition:", d["links"]["html"])
