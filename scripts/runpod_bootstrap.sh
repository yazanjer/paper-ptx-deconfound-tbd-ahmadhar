#!/usr/bin/env bash
# Entry point for a RunPod GPU pod. Idempotent: safe to restart; finished runs are skipped.
#
# Required pod environment variables (set them as RunPod *secrets*, never in git):
#   GITHUB_TOKEN     fine-grained PAT: repo yazanjer/ptx-deconfound, Contents read/write
#   KAGGLE_USERNAME  KAGGLE_KEY     (SIIM-ACR + NIH ChestX-ray14 downloads)
# Optional:
#   CANDID_URL       download link emailed after the CANDID-PTX DUA is approved
#   STAGES           default "data gate1_nih"   (e.g. "gate1", "tube_labels main", "gate2")
#   MAX_HOURS        wall-clock budget for the queue (default 10)
#   STOP_WHEN_DONE   1 (default) -> the pod stops itself when finished, so billing stops
set -uo pipefail
STAGES="${STAGES:-data gate1_nih}"
MAX_HOURS="${MAX_HOURS:-10}"
WS=/workspace
RAW=$WS/raw
DATA=$WS/data
REPO=$WS/ptx-deconfound
mkdir -p $RAW $DATA
export GIT_TERMINAL_PROMPT=0
unz() { python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }  # image has no unzip
exec > >(tee -a $WS/bootstrap.log) 2>&1
echo "=== bootstrap $(date -u) stages=[$STAGES] budget=${MAX_HOURS}h"

finish() {
  echo "=== finished $(date -u)"
  if [ -d $REPO/.git ]; then  # publish the log so progress/failures are visible from GitHub
    mkdir -p $REPO/results/logs && cp $WS/bootstrap.log $REPO/results/logs/bootstrap_${RUNPOD_POD_ID:-local}.log
    (cd $REPO && bash scripts/push_results.sh "bootstrap log ${RUNPOD_POD_ID:-local}") || true
  fi
  if [ "${STOP_WHEN_DONE:-1}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
  fi
}
trap finish EXIT

# ---- code
if [ ! -d $REPO/.git ]; then
  git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/yazanjer/ptx-deconfound.git" $REPO
fi
cd $REPO && { git rebase --abort 2>/dev/null; rm -rf .git/rebase-merge .git/rebase-apply
  mkdir -p $WS/keep && [ -d results/runs ] && cp -rn results/runs $WS/keep/   # keep finished runs not yet pushed
  git fetch -q origin main && git reset -q --hard origin/main
  [ -d $WS/keep/runs ] && mkdir -p results/runs && cp -rn $WS/keep/runs/. results/runs/; }
# fail fast if the token cannot push (otherwise hours of results would be lost)
mkdir -p results/logs && date -u > results/logs/start_${RUNPOD_POD_ID:-local}.txt
git add results/logs && git -c user.name=ptx-runner -c user.email=ptx-runner@users.noreply.github.com commit -q -m "pod start ${RUNPOD_POD_ID:-local}"
if ! git push -q origin HEAD:main; then echo "FATAL: GitHub push failed - check github_token has Contents read/write"; exit 1; fi
echo "github push OK"
# heartbeat: publish the running log to GitHub every 10 minutes
( while true; do sleep 600; mkdir -p results/logs; cp $WS/bootstrap.log results/logs/bootstrap_${RUNPOD_POD_ID:-local}.log
  bash scripts/push_results.sh "heartbeat ${RUNPOD_POD_ID:-local}" >/dev/null 2>&1; done ) &
pip install -q -r requirements.txt

# ---- data (each dataset is prepared once and cached on the network volume)
if [[ " $STAGES " == *" data "* ]]; then
  mkdir -p ~/.kaggle && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  if [ ! -f $DATA/ptx498/manifest.csv ]; then
    curl -sL -o $RAW/ptx498.zip "https://zenodo.org/api/records/8266529/files/PTX-498-v2-fix.zip/content"
    unz $RAW/ptx498.zip $RAW/ptx498 && python -m ptx.data.prepare ptx498 --raw $RAW/ptx498 --out $DATA && rm -rf $RAW/ptx498 $RAW/ptx498.zip
  fi
  if [ ! -f $DATA/nih/manifest.csv ]; then
    curl -sL -o $RAW/nih_cohort.zip "https://zenodo.org/api/records/22839698/files/Pneumothorax_Zenodo_READY_v1.0.0.zip/content"
    unz $RAW/nih_cohort.zip $RAW/nih_cohort
    have=$(python - <<PY
import glob, os, pandas as pd
ids = set(pd.read_csv(glob.glob("$RAW/nih_cohort/**/cohort_manifest.csv", recursive=True)[0]).image_id)
got = {os.path.basename(p) for p in glob.glob("$RAW/nih/**/*.png", recursive=True)}
print(len(ids - got))
PY
)
    echo "NIH cohort images missing from raw folder: $have"
    if [ "$have" != "0" ]; then
      kaggle datasets download -d nih-chest-xrays/data -p $RAW/nih --unzip
    fi
    python -m ptx.data.prepare nih --raw $RAW/nih --cohort $RAW/nih_cohort --out $DATA && rm -rf $RAW/nih  # keep only the 4,388 cohort images
  fi
  if [ ! -f $DATA/siim/manifest.csv ]; then
    [ -n "$(find $RAW/siim -name '*.dcm' 2>/dev/null | head -1)" ] || kaggle datasets download -d jesperdramsch/siim-acr-pneumothorax-segmentation-data -p $RAW/siim --unzip
    python -m ptx.data.prepare siim --raw $RAW/siim --out $DATA && rm -rf $RAW/siim
  fi
  if [ ! -f $DATA/candid/manifest.csv ] && [ -n "${CANDID_URL:-}" ]; then
    curl -sL -o $RAW/candid.zip "$CANDID_URL" && unz $RAW/candid.zip $RAW/candid
    python -m ptx.data.prepare candid --raw $RAW/candid  --out $DATA  # raw kept: re-run with --rle-mode absolute if overlays look wrong
  fi
  python scripts/data_report.py --data $DATA --out results/data_report.md
  # splits are created once (with CANDID as dev) and then frozen in git
  if [ -f $DATA/candid/manifest.csv ] && [ ! -f splits/candid.csv ]; then
    python -m ptx.data.splits --data $DATA --dev candid --out splits
    python scripts/check_masks.py --data $DATA --out results/mask_checks
  elif [ ! -f splits/nih.csv ] && [ -f $DATA/nih/manifest.csv ]; then
    # before CANDID arrives: only the NIH released split is needed for the Gate-1 NIH pilot
    python -m ptx.data.splits --data $DATA --dev none --externals nih --out splits
  fi
  bash scripts/push_results.sh "data prepared"
fi

# ---- experiment stages
RUN_STAGES=$(echo "$STAGES" | sed 's/\bdata\b//g')
if [ -n "$(echo $RUN_STAGES | tr -d ' ')" ]; then
  python scripts/run_queue.py --stage $RUN_STAGES --data $DATA --hours "$MAX_HOURS" --push
fi
python scripts/upload_weights.py --runs results/runs || true
