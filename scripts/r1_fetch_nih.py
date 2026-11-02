"""Download only the NIH ChestX-ray14 PNGs of the device-cohort test split (and Data_Entry_2017.csv) from Kaggle.

    python scripts/r1_fetch_nih.py --raw /workspace/raw/nih --splits splits
"""
import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DS = "nih-chest-xrays/data"
# last image of each archive folder in NIH ChestX-ray14 (images_001 ... images_012)
UPPER = ["00001335_006", "00003923_013", "00006585_006", "00009232_003", "00011558_007", "00013774_025",
         "00016051_009", "00018387_034", "00020945_049", "00024717_000", "00028173_002", "99999999_999"]

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--splits", type=Path, default=Path("splits"))
ap.add_argument("--split", default="test")
a = ap.parse_args()
a.raw.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()


def get(path: str, dest: Path) -> bool:
    for attempt in range(4):
        try:
            api.dataset_download_file(DS, path, path=str(dest), quiet=True)
            break
        except BaseException as e:  # kaggle may raise SystemExit on HTTP errors (e.g. 429)
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():
                return False
            time.sleep(5 * (attempt + 1))
    else:
        return False
    name = Path(path).name
    z = dest / (name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        z.unlink()
    return (dest / name).exists()


ids = pd.read_csv(a.splits / "nih.csv").query("split == @a.split").image_id.tolist()


def one(iid):
    if (a.raw / f"{iid}.png").exists():
        return iid, True
    first = next(i for i, u in enumerate(UPPER) if iid <= u)
    order = [first] + [i for i in (first + 1, first - 1) if 0 <= i < 12] + [i for i in range(12) if abs(i - first) > 1]
    for k in order:
        if get(f"images_{k + 1:03d}/images/{iid}.png", a.raw):
            return iid, True
    return iid, False


meta = a.raw.parent / "Data_Entry_2017.csv"
if not meta.exists():
    get("Data_Entry_2017.csv", a.raw.parent)
print("metadata:", meta.exists(), flush=True)
res = []
for k, iid in enumerate(ids):
    res.append(one(iid))
    if k % 50 == 0:
        print(f"nih fetch {k}/{len(ids)} ok={sum(r[1] for r in res)}", flush=True)
miss = [i for i, ok in res if not ok]
print(f"nih {a.split}: {len(ids) - len(miss)}/{len(ids)} images; missing {miss[:10]}")
