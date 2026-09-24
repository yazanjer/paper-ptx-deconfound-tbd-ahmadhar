"""Gate 2 helper: wait until every expected run is finished on GitHub (runs are trained on several pods).

Syncs with origin every few minutes (via push_results.sh, which also publishes this pod's files) and
exits 0 once the number of `results/runs/<prefix>*/seed*/DONE` markers reaches --expect.

    python scripts/wait_runs.py --expect 56 --prefix pb_ --hours 5
"""
import argparse
import glob
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--expect", type=int, required=True)
ap.add_argument("--prefix", default="pb_")
ap.add_argument("--hours", type=float, default=5)
ap.add_argument("--every", type=int, default=300, help="seconds between checks")
a = ap.parse_args()
t0 = time.time()
while True:
    subprocess.run("bash scripts/push_results.sh 'gate2: waiting for runs'", shell=True, check=False)
    n = len(glob.glob(f"results/runs/{a.prefix}*/seed*/DONE"))
    print(f"{time.strftime('%H:%M:%S')} finished runs: {n}/{a.expect}", flush=True)
    if n >= a.expect:
        sys.exit(0)
    if time.time() - t0 > a.hours * 3600:
        sys.exit(f"gave up after {a.hours} h with {n}/{a.expect} runs")
    time.sleep(a.every)
