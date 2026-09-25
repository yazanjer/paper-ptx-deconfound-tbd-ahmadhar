"""Upload the trained model checkpoints to a Zenodo deposition (created as an unpublished DRAFT).

Every finished run (results/runs/<run>/seed<k>/best.pt, pilots excluded) is uploaded as
<run>__seed<k>.pt, together with MODEL_CARD.md and a manifest (run, seed, size, SHA-256, MD5,
decision threshold). Re-running is safe: the deposition id is kept in results/zenodo/deposition.json
and files already on Zenodo with the same MD5 are skipped.

Publishing is NOT done here: a published Zenodo record is permanent and public. Review the draft
on zenodo.org and press "Publish" (or run with --publish) when the repository goes public.

    ZENODO_TOKEN=... python scripts/upload_zenodo.py --runs results/runs [--sandbox] [--publish]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_URL = "https://github.com/yazanjer/ptx-deconfound"
TITLE = ("ptx-deconfound: trained model checkpoints for untreated-pneumothorax detection "
         "(12 configurations, 54 runs)")
DESCRIPTION = """
<p>Trained PyTorch checkpoints for the study <i>"Adversarial deconfounding with pseudo device labels does not
remove the device shortcut in pneumothorax detection: a pre-registered, multi-seed external evaluation of
twelve models"</i>.</p>
<p>Contains one checkpoint per training run (12 model configurations, 3&ndash;5 seeds each, 54 files):
the proposed U-Net/EfficientNet-B4 model with gradient-reversal device adversary and top-k mask pooling,
its ablations and sensitivity variants, and re-trained baselines (U-Net ResNet34, SegFormer-B2, Group DRO,
auxiliary device classifier, RAD-DINO linear probe and fine-tuned). All were trained on SIIM-ACR with the
frozen splits in the code repository. Each checkpoint is the epoch selected on SIIM-ACR validation; the
matching 95%-specificity decision threshold is listed in <code>manifest.csv</code>.</p>
<p>Load with the code at <a href="{repo}">{repo}</a>:
<code>m = build_model(yaml.safe_load(open(config))); m.load_state_dict(torch.load(ckpt, map_location="cpu"))</code>.
See MODEL_CARD.md for intended use and limitations. Not a medical device; research use only.</p>
""".format(repo=REPO_URL)


def digests(p: Path):
    s, m = hashlib.sha256(), hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            s.update(chunk)
            m.update(chunk)
    return s.hexdigest(), m.hexdigest()


def model_card(rows):
    runs = sorted({r["run"] for r in rows})
    lines = ["# Model card: ptx-deconfound checkpoints", "",
             "**Task.** Image-level pneumothorax detection (and, for segmentation models, pixel masks) on frontal chest radiographs.",
             "**Training data.** SIIM-ACR (development split in the code repository); device labels on SIIM-ACR are pseudo-labels.",
             "**Evaluation.** NIH ChestX-ray14-derived device cohort (primary external) and PTX-498 (secondary); see the paper.",
             "**Key limitation.** Every configuration relies partly on visible devices (a device shortcut): performance on",
             "untreated pneumothorax is markedly lower than on device-present cases, and decision thresholds do not",
             "transfer between datasets without recalibration. Research use only; not for clinical decisions.", "",
             f"**Contents.** {len(rows)} checkpoints from {len(runs)} configurations: " + ", ".join(runs) + ".",
             "File name = `<configuration>__seed<k>.pt`; architecture and hyper-parameters are in",
             f"`results/runs/<configuration>/seed<k>/config.yaml` of {REPO_URL}.", "",
             "**Licence.** CC-BY-4.0 for the weights; the code is MIT. Dataset licences of SIIM-ACR and NIH ChestX-ray14 apply."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/zenodo"))
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", action="store_true", help="publish the record (permanent, public)")
    a = ap.parse_args()
    base = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    tok = os.environ.get("ZENODO_TOKEN", "").strip()
    if not tok:
        sys.exit("ZENODO_TOKEN is not set (RunPod secret 'zenodo')")
    S = requests.Session()
    S.headers["Authorization"] = f"Bearer {tok}"
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- collect checkpoints and write manifest + model card
    rows = []
    for ck in sorted(glob.glob(str(a.runs / "*" / "seed*" / "best.pt"))):
        ck = Path(ck)
        run, seed = ck.parent.parent.name, ck.parent.name
        if run.startswith("pilot_"):
            continue
        thr = ck.parent / "threshold.json"
        sha, md5 = digests(ck)
        rows.append({"file": f"{run}__{seed}.pt", "run": run, "seed": seed.replace("seed", ""),
                     "bytes": ck.stat().st_size, "sha256": sha, "md5": md5,
                     "threshold_spec95": json.load(open(thr)).get("threshold_spec95") if thr.exists() else "",
                     "path": str(ck)})
    if not rows:
        sys.exit("no checkpoints found; run scripts/fetch_weights.py first")
    print(f"{len(rows)} checkpoints, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB")
    man = a.out / "manifest.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "path"])
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k != "path"} for r in rows)
    card = a.out / "MODEL_CARD.md"
    card.write_text(model_card(rows))

    # ---- create or reuse the draft deposition
    state = a.out / ("deposition_sandbox.json" if a.sandbox else "deposition.json")
    dep = None
    if state.exists():
        did = json.load(open(state))["id"]
        r = S.get(f"{base}/deposit/depositions/{did}", timeout=60)
        if r.ok and not r.json().get("submitted"):
            dep = r.json()
    if dep is None:
        meta = {"metadata": {
            "title": TITLE, "upload_type": "dataset", "description": DESCRIPTION,
            "creators": [{"name": "Aljeroudi, Yazan"}], "access_right": "open", "license": "cc-by-4.0",
            "keywords": ["pneumothorax", "chest radiograph", "deep learning", "model weights", "shortcut learning",
                         "domain-adversarial training", "negative result"],
            "related_identifiers": [{"identifier": REPO_URL, "relation": "isSupplementTo", "resource_type": "software"}],
            "notes": "Checkpoints only; images are not redistributed. See MODEL_CARD.md and manifest.csv."}}
        r = S.post(f"{base}/deposit/depositions", json=meta, timeout=60)
        if not r.ok:
            sys.exit(f"could not create deposition: {r.status_code} {r.text[:500]}")
        dep = r.json()
        print("created draft deposition", dep["id"])
    json.dump({"id": dep["id"], "html": dep["links"].get("html"), "bucket": dep["links"]["bucket"],
               "created": dep.get("created")}, open(state, "w"), indent=1)

    # ---- upload files (skip ones already present with the same md5)
    have = {f["filename"]: f.get("checksum", "").removeprefix("md5:") for f in
            S.get(f"{base}/deposit/depositions/{dep['id']}/files", timeout=60).json()}
    bucket = dep["links"]["bucket"]
    todo = [(Path(r["path"]), r["file"], r["md5"]) for r in rows]
    todo += [(man, "manifest.csv", digests(man)[1]), (card, "MODEL_CARD.md", digests(card)[1])]
    for i, (p, name, md5) in enumerate(todo, 1):
        if have.get(name) == md5:
            print(f"[{i}/{len(todo)}] {name} already uploaded")
            continue
        for attempt in range(5):
            try:
                t0 = time.time()
                with open(p, "rb") as fh:
                    r = S.put(f"{bucket}/{name}", data=fh, timeout=3600)
                if r.ok and r.json().get("checksum", "").removeprefix("md5:") == md5:
                    mb = p.stat().st_size / 1e6
                    print(f"[{i}/{len(todo)}] {name} {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)
                    break
                print(f"[{i}/{len(todo)}] {name} failed ({r.status_code}): {r.text[:200]}")
            except requests.RequestException as ex:
                print(f"[{i}/{len(todo)}] {name} error: {ex}")
            time.sleep(20 * (attempt + 1))
        else:
            sys.exit(f"giving up on {name}")

    files = S.get(f"{base}/deposit/depositions/{dep['id']}/files", timeout=60).json()
    print(f"deposition {dep['id']}: {len(files)} files, {sum(f['filesize'] for f in files) / 1e9:.2f} GB")
    if a.publish:
        r = S.post(f"{base}/deposit/depositions/{dep['id']}/actions/publish", timeout=120)
        if not r.ok:
            sys.exit(f"publish failed: {r.status_code} {r.text[:500]}")
        rec = r.json()
        print("published:", rec.get("doi"), rec["links"].get("html"))
        json.dump({**json.load(open(state)), "doi": rec.get("doi"), "published": True}, open(state, "w"), indent=1)
    else:
        print("DRAFT ready for review (not published):", dep["links"].get("html"))


if __name__ == "__main__":
    main()
