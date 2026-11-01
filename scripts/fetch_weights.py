"""Download best.pt from the `weights-v1` release for every finished run that lacks it locally.

Runs are trained on several pods; Gate 2 evaluates all of them on one pod, so the weights trained
elsewhere are fetched from the release assets (asset name = <run>__<seed>.pt). Needs GITHUB_TOKEN.

    python scripts/fetch_weights.py --runs results/runs
"""
import argparse
import glob
import hashlib
import os
from pathlib import Path

import requests

REPO = os.environ.get("GH_REPO", "yazanjer/paper-ptx-deconfound-tbd-ahmadhar")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path("results/runs"))
ap.add_argument("--tag", default="weights-v1")
a = ap.parse_args()
H = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}
rel = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}", headers=H, timeout=60).json()
assets, page = {}, 1
while True:
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/{rel['id']}/assets", headers=H,
                     params={"per_page": 100, "page": page}, timeout=60).json()
    if not r:
        break
    assets.update({x["name"]: x for x in r})
    page += 1
missing = []
for done in sorted(glob.glob(str(a.runs / "*" / "seed*" / "DONE"))):
    od = Path(done).parent
    if od.parent.name.startswith("pilot_") or (od / "best.pt").exists():
        continue
    name = f"{od.parent.name}__{od.name}.pt"
    if name not in assets:
        missing.append(name)
        continue
    x = assets[name]
    tmp = od / "best.pt.part"
    with requests.get(x["url"], headers={**H, "Accept": "application/octet-stream"}, stream=True, timeout=600) as rr:
        rr.raise_for_status()
        h = hashlib.sha256()
        with open(tmp, "wb") as fh:
            for chunk in rr.iter_content(1 << 20):
                fh.write(chunk)
                h.update(chunk)
    want = (x.get("digest") or "").removeprefix("sha256:")
    if want and h.hexdigest() != want:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for {name}")
    tmp.rename(od / "best.pt")
    print("fetched", name)
if missing:
    raise SystemExit(f"no release asset for: {missing}")
print("all weights present")
