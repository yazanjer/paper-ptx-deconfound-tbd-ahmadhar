"""Resumable experiment queue. Skips finished runs (DONE marker), pushes results to GitHub after
each run, stops cleanly before a wall-clock budget runs out.

    python scripts/run_queue.py --stage gate1 --data /workspace/data --hours 10 [--push]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptx.train import load_config, run  # noqa: E402

AVG_RUN_HOURS = 1.2  # conservative estimate used to avoid starting a run that cannot finish


def push(msg):
    subprocess.run("bash scripts/push_results.sh " + repr(msg), shell=True, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, nargs="+")
    ap.add_argument("--data", default="/workspace/data")
    ap.add_argument("--hours", type=float, default=1e9)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/runs"))
    a = ap.parse_args()
    E = yaml.safe_load(open("configs/experiments.yaml"))
    t0 = time.time()
    failed = []
    for stage in a.stage:
        for item in E["stages"][stage]:
            left = a.hours - (time.time() - t0) / 3600
            if "command" in item:
                cmd = item["command"].format(data=a.data)
                print(f"== [{stage}] {cmd}")
                r = subprocess.run(cmd, shell=True)
                if a.push:
                    push(f"[{stage}] {cmd.split()[2] if len(cmd.split()) > 2 else cmd}")
                if r.returncode != 0:
                    sys.exit(f"command failed: {cmd}")
                continue
            seeds = item["seeds"] if isinstance(item["seeds"], list) else E[item["seeds"]]
            cfg = load_config(f"configs/runs/{item['config']}.yaml")
            cfg["data"]["root"] = a.data
            for s in seeds:
                if (a.out / cfg["name"] / f"seed{s}" / "DONE").exists():
                    continue
                if left < AVG_RUN_HOURS:
                    print(f"time budget nearly spent ({left:.2f} h left); stopping before {cfg['name']} seed {s}")
                    if a.push:
                        push(f"[{stage}] partial: budget reached")
                    return
                print(f"== [{stage}] {cfg['name']} seed {s}")
                try:
                    run(cfg, s, a.out)
                except Exception:  # one broken config must not stop the rest of the queue
                    traceback.print_exc()
                    failed.append(f"{cfg['name']} seed {s}")
                    print(f"!! [{stage}] {cfg['name']} seed {s} FAILED; continuing")
                if a.push:
                    push(f"[{stage}] {cfg['name']} seed {s}")
                left = a.hours - (time.time() - t0) / 3600
    print("queue finished" + (f"; FAILED runs: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
