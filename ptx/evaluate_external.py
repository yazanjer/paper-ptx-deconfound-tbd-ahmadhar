"""Gate 2: open the external test sets ONCE for every finished run (reuses best.pt, no training).

    PTX_UNLOCK_EXTERNAL=1 python -m ptx.evaluate_external --runs results/runs
Refuses to run unless PREREGISTRATION.md is committed and results/prereg_hash.txt matches it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from pathlib import Path

import torch
import yaml

from ptx.models.net import build_model
from ptx.train import evaluate_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    h = hashlib.sha256(Path("PREREGISTRATION.md").read_bytes()).hexdigest()
    frozen = Path("results/prereg_hash.txt")
    if not frozen.exists() or frozen.read_text().strip() != h:
        raise SystemExit("PREREGISTRATION.md changed or not frozen: run scripts/freeze_prereg.sh first")
    os.environ["PTX_UNLOCK_EXTERNAL"] = "1"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for done in sorted(glob.glob(str(a.runs / "*" / "seed*" / "DONE"))):
        od = Path(done).parent
        if a.only and od.parent.name not in a.only:
            continue
        if (od / "EXTERNAL_DONE").exists() or not (od / "best.pt").exists():
            continue
        cfg = yaml.safe_load(open(od / "config.yaml"))
        m = build_model(cfg).to(dev)
        m.load_state_dict(torch.load(od / "best.pt", map_location=dev))
        cfg["eval"]["sets"] = [s for s in cfg["eval"]["sets"] if s.get("external")]
        evaluate_sets(m, cfg, od, dev, cfg["train"].get("workers", 8))
        (od / "EXTERNAL_DONE").write_text(h)
        print("external done:", od)


if __name__ == "__main__":
    main()
