"""Convert raw datasets into one common on-disk format.

Output per dataset, under <out>/<dataset>/:
    img/<image_id>.png    uint8 grayscale, resized to `size` x `size`
    ptx/<image_id>.png    uint8 {0,255} pneumothorax mask (if available)
    tube/<image_id>.png   uint8 {0,255} chest-tube mask (if available)
    manifest.csv          one row per image (columns below)

Manifest columns
    image_id, dataset, patient_id, ptx (0/1), tube (1/0/-1 unknown),
    device (1/0/-1: any visible device), treated (1/0/-1), treated_source,
    has_ptx_mask, has_tube_mask, spacing_mm (at output size, NaN if unknown),
    site, view, source_split

`treated` is the stratification variable used everywhere: chest tube if known,
otherwise any visible device (NIH cohort), otherwise unknown.

Usage
    python -m ptx.data.prepare siim   --raw /workspace/raw/siim   --out /workspace/data
    python -m ptx.data.prepare candid --raw /workspace/raw/candid --out /workspace/data
    python -m ptx.data.prepare nih    --raw /workspace/raw/nih    --cohort /workspace/raw/nih_cohort --out /workspace/data
    python -m ptx.data.prepare ptx498 --raw /workspace/raw/ptx498 --out /workspace/data
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .rle import union_rles

COLS = ["image_id", "dataset", "patient_id", "ptx", "tube", "device", "treated", "treated_source",
        "has_ptx_mask", "has_tube_mask", "spacing_mm", "site", "view", "source_split"]


def _resize_img(a: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)


def _resize_mask(m: np.ndarray, size: int) -> np.ndarray:
    r = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (r >= 0.5).astype(np.uint8) * 255


def _to_uint8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def _read_dicom(path: str):
    import pydicom
    ds = pydicom.dcmread(path)
    a = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    spacing = np.nan
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if hasattr(ds, tag):
            try:
                spacing = float(getattr(ds, tag)[0])
                break
            except Exception:
                pass
    return ds, _to_uint8(a), spacing


def _row(**kw):
    r = {c: np.nan for c in COLS}
    r.update(kw)
    return r


def _write(out_dir: Path, image_id: str, img, ptx=None, tube=None, size=512):
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "img" / f"{image_id}.png"), _resize_img(img, size))
    if ptx is not None:
        (out_dir / "ptx").mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "ptx" / f"{image_id}.png"), _resize_mask(ptx, size))
    if tube is not None:
        (out_dir / "tube").mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "tube" / f"{image_id}.png"), _resize_mask(tube, size))


def _find_csv(raw: Path, *keywords):
    for p in sorted(raw.rglob("*.csv")):
        n = p.name.lower()
        if all(k in n for k in keywords):
            return p
    return None


def _id_rle_cols(df: pd.DataFrame):
    cols = {c.strip(): c for c in df.columns}
    id_col = next((cols[c] for c in cols if c.lower() in ("imageid", "sopinstanceuid", "image_id", "id")), df.columns[0])
    rle_col = next((cols[c] for c in cols if "encoded" in c.lower() or "rle" in c.lower()), df.columns[1])
    return id_col, rle_col


# --------------------------------------------------------------------------- SIIM-ACR
def prepare_siim(raw: Path, out: Path, size: int, rle_mode: str):
    """Kaggle dataset `jesperdramsch/siim-acr-pneumothorax-segmentation-data`:
    dicom-images-train/**.dcm + train-rle.csv (relative RLE, one row per lesion)."""
    csv = _find_csv(raw, "train", "rle") or _find_csv(raw, "train")
    df = pd.read_csv(csv)
    id_col, rle_col = _id_rle_cols(df)
    groups = df.groupby(id_col)[rle_col].apply(list).to_dict()
    dcm = {Path(p).stem: p for p in glob.glob(str(raw / "**" / "*.dcm"), recursive=True)}
    od = out / "siim"
    rows = []
    for iid, rles in tqdm(groups.items(), desc="siim"):
        if iid not in dcm:
            continue
        ds, img, sp = _read_dicom(dcm[iid])
        h, w = img.shape
        m = union_rles(rles, h, w, rle_mode)
        _write(od, iid, img, ptx=m, size=size)
        rows.append(_row(image_id=iid, dataset="siim", patient_id=str(getattr(ds, "PatientID", iid)),
                         ptx=int(m.any()), tube=-1, device=-1, treated=-1, treated_source="pseudo_pending",
                         has_ptx_mask=1, has_tube_mask=0, spacing_mm=sp * w / size if sp == sp else np.nan,
                         site="siim", view=str(getattr(ds, "ViewPosition", "")), source_split="train"))
    pd.DataFrame(rows, columns=COLS).to_csv(od / "manifest.csv", index=False)
    print(f"siim: {len(rows)} images, {sum(r['ptx'] for r in rows)} positive (csv rows {len(groups)}, dicoms {len(dcm)})")


# --------------------------------------------------------------------------- CANDID-PTX
def prepare_candid(raw: Path, out: Path, size: int, rle_mode: str):
    """CANDID-PTX (Univ. Auckland): DICOMs + CSVs for pneumothorax, chest tube, rib fracture.
    CSV column names are detected automatically; run scripts/check_masks.py afterwards."""
    ptx_csv = _find_csv(raw, "pneumothorax")
    tube_csv = _find_csv(raw, "tube")
    assert ptx_csv is not None, f"no pneumothorax csv under {raw}"
    pdf = pd.read_csv(ptx_csv)
    pid, prle = _id_rle_cols(pdf)
    ptx_rles = pdf.groupby(pid)[prle].apply(list).to_dict()
    tube_rles = {}
    if tube_csv is not None:
        tdf = pd.read_csv(tube_csv)
        tid, trle = _id_rle_cols(tdf)
        tube_rles = tdf.groupby(tid)[trle].apply(list).to_dict()
    od = out / "candid"
    rows = []
    files = glob.glob(str(raw / "**" / "*.dcm"), recursive=True)
    for f in tqdm(files, desc="candid"):
        ds, img, sp = _read_dicom(f)
        uid = str(getattr(ds, "SOPInstanceUID", Path(f).stem))
        if uid not in ptx_rles and Path(f).stem in ptx_rles:
            uid = Path(f).stem
        h, w = img.shape
        pm = union_rles(ptx_rles.get(uid, []), h, w, rle_mode)
        tm = union_rles(tube_rles.get(uid, []), h, w, rle_mode) if tube_csv is not None else None
        tube = int(tm.any()) if tm is not None else -1
        _write(od, uid, img, ptx=pm, tube=tm, size=size)
        rows.append(_row(image_id=uid, dataset="candid", patient_id=str(getattr(ds, "PatientID", "") or uid),
                         ptx=int(pm.any()), tube=tube, device=-1, treated=tube, treated_source="tube_mask",
                         has_ptx_mask=1, has_tube_mask=int(tm is not None), spacing_mm=sp * w / size if sp == sp else np.nan,
                         site="candid", view=str(getattr(ds, "ViewPosition", "")), source_split=""))
    pd.DataFrame(rows, columns=COLS).to_csv(od / "manifest.csv", index=False)
    print(f"candid: {len(rows)} images, {sum(r['ptx'] for r in rows)} PTX, {sum(r['tube'] == 1 for r in rows)} with tube")


# --------------------------------------------------------------------------- NIH device cohort
def prepare_nih(raw: Path, cohort: Path, out: Path, size: int):
    """NIH ChestX-ray14 images (e.g. Kaggle `nih-chest-xrays/data`) + Zenodo 22839698 cohort manifest."""
    man = pd.read_csv(next(cohort.rglob("cohort_manifest.csv")))
    pngs = {Path(p).name: p for p in glob.glob(str(raw / "**" / "*.png"), recursive=True)}
    od = out / "nih"
    rows, missing = [], 0
    dev_map = {"with_visible_device": 1, "without_visible_device": 0, "uncertain": -1}
    for r in tqdm(man.itertuples(), total=len(man), desc="nih"):
        if r.image_id not in pngs:
            missing += 1
            continue
        img = cv2.imread(pngs[r.image_id], cv2.IMREAD_GRAYSCALE)
        iid = Path(r.image_id).stem
        _write(od, iid, img, size=size)
        dev = dev_map.get(r.visible_device_status, -1)
        rows.append(_row(image_id=iid, dataset="nih", patient_id=r.patient_key, ptx=int(r.class_label == "pneumothorax"),
                         tube=-1, device=dev, treated=dev, treated_source="any_visible_device",
                         has_ptx_mask=0, has_tube_mask=0, spacing_mm=np.nan,  # NIH PNGs carry no spacing
                         site="nih", view=r.view_position, source_split=r.split))
    pd.DataFrame(rows, columns=COLS).to_csv(od / "manifest.csv", index=False)
    print(f"nih: {len(rows)} images written, {missing} missing from raw folder")


# --------------------------------------------------------------------------- PTX-498
def prepare_ptx498(raw: Path, out: Path, size: int):
    """Zenodo 8266529 (v2-fix): Site{A,B,C}/<n>.1.img.png and <n>.2.mask.png. Positives only."""
    od = out / "ptx498"
    rows = []
    for img_p in tqdm(sorted(glob.glob(str(raw / "**" / "*.1.img.png"), recursive=True)), desc="ptx498"):
        site = Path(img_p).parent.name
        n = Path(img_p).name.split(".")[0]
        mask_p = img_p.replace(".1.img.png", ".2.mask.png")
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        m = (cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        iid = f"{site}_{n}"
        _write(od, iid, img, ptx=m, size=size)
        rows.append(_row(image_id=iid, dataset="ptx498", patient_id=iid, ptx=int(m.any()), tube=-1, device=-1,
                         treated=-1, treated_source="unknown", has_ptx_mask=1, has_tube_mask=0, spacing_mm=np.nan,
                         site=site, view="", source_split=""))
    pd.DataFrame(rows, columns=COLS).to_csv(od / "manifest.csv", index=False)
    print(f"ptx498: {len(rows)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["siim", "candid", "nih", "ptx498"])
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cohort", type=Path, help="NIH: folder with the unzipped Zenodo cohort package")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rle-mode", default="relative", choices=["relative", "absolute"])
    a = ap.parse_args()
    if a.dataset == "siim":
        prepare_siim(a.raw, a.out, a.size, a.rle_mode)
    elif a.dataset == "candid":
        prepare_candid(a.raw, a.out, a.size, a.rle_mode)
    elif a.dataset == "nih":
        prepare_nih(a.raw, a.cohort, a.out, a.size)
    else:
        prepare_ptx498(a.raw, a.out, a.size)


if __name__ == "__main__":
    main()
