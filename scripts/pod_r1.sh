#!/usr/bin/env bash
# R1 revision job (IEEE Access resubmission): inference-only analyses on the released checkpoints.
# Runs on a single GPU pod; reuses /workspace/data/siim if an earlier pod prepared it. No training.
# Env: GITHUB_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, MAX_HOURS (hard stop), BRANCH (code), OUT_BRANCH (results).
export GIT_AUTHOR_NAME=ptx-runner GIT_COMMITTER_NAME=ptx-runner GIT_AUTHOR_EMAIL=ptx-runner@users.noreply.github.com GIT_COMMITTER_EMAIL=ptx-runner@users.noreply.github.com
set -uo pipefail
export GIT_TERMINAL_PROMPT=0 OMP_NUM_THREADS=4 PYTHONPATH=.
GH_REPO=${GH_REPO:-yazanjer/paper-ptx-deconfound-tbd-ahmadhar}; export GH_REPO
BRANCH=${BRANCH:-r1-ieee}; OUT_BRANCH=${OUT_BRANCH:-r1-results}
WS=/workspace; RAW=$WS/raw; DATA=$WS/data; R=$WS/r1code; LOG=$WS/r1.log
mkdir -p $RAW $DATA
exec > >(tee -a $LOG) 2>&1
echo "=== r1 start $(date -u) pod=${RUNPOD_POD_ID:-local} max=${MAX_HOURS:-6}h"
( sleep $(( ${MAX_HOURS:-6} * 3600 )); echo "=== hard stop"; runpodctl stop pod "$RUNPOD_POD_ID" ) &
stopnow() { echo "=== stop: $1 $(date -u)"; push "final: $1"; sleep 120; runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity; }
unz() { python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
fetch_zip() { for i in 1 2 3 4 5; do curl -sL -o "$2" "$1" && python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$2" && return 0; echo "retry $1"; sleep 45; done; return 1; }
push() {  # results go to their own branch so the code branch is never touched by the pod
  [ -d $R/.git ] || return 0
  ( cd $R && mkdir -p results/r1 && cp $LOG results/r1/pod_log.txt && git add -f results/r1 && { git diff --cached --quiet || git commit -qm "r1 results: $1"; } \
    && git push -qf origin HEAD:refs/heads/$OUT_BRANCH && echo "pushed: $1" ) || echo "push failed (continuing)"
}

# ---- code
if [ ! -d $R/.git ]; then
  git clone -q -b $BRANCH "https://x-access-token:${GITHUB_TOKEN}@github.com/${GH_REPO}.git" $R || { echo "FATAL clone"; sleep 600; runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity; }
fi
cd $R && git checkout -q -B r1-pod
pip install -q -r requirements.txt fvcore 2>&1 | tail -2
python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0))" || stopnow "no gpu"
mkdir -p ~/.kaggle && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

# ---- data
if [ ! -f $DATA/siim/manifest.csv ]; then
  echo "preparing SIIM"
  [ -n "$(find $RAW/siim -name '*.dcm' 2>/dev/null | head -1)" ] || kaggle datasets download -d jesperdramsch/siim-acr-pneumothorax-segmentation-data -p $RAW/siim --unzip -q
  python -m ptx.data.prepare siim --raw $RAW/siim --out $DATA && rm -rf $RAW/siim || stopnow "siim prepare"
else echo "SIIM cached on volume: $(wc -l < $DATA/siim/manifest.csv) rows"; fi
if [ ! -f $DATA/ptx498/manifest.csv ]; then
  fetch_zip "https://zenodo.org/api/records/8266529/files/PTX-498-v2-fix.zip/content" $RAW/ptx498.zip && unz $RAW/ptx498.zip $RAW/ptx498 \
    && python -m ptx.data.prepare ptx498 --raw $RAW/ptx498 --out $DATA && rm -rf $RAW/ptx498 $RAW/ptx498.zip || echo "WARN ptx498 failed"
fi
if [ ! -f $DATA/nih/manifest.csv ]; then
  fetch_zip "https://zenodo.org/api/records/22839698/files/Pneumothorax_Zenodo_READY_v1.0.0.zip/content" $RAW/nih_cohort.zip && unz $RAW/nih_cohort.zip $RAW/nih_cohort || echo "WARN nih cohort zip"
  python scripts/r1_fetch_nih.py --raw $RAW/nih --splits splits --split test
  python -m ptx.data.prepare nih --raw $RAW/nih --cohort $RAW/nih_cohort --out $DATA || echo "WARN nih prepare failed"
fi
mkdir -p results/r1
if [ -f $RAW/Data_Entry_2017.csv ]; then
  python - <<'PY'
import pandas as pd
m = pd.read_csv('/workspace/raw/Data_Entry_2017.csv')
m['image_id'] = m['Image Index'].str.replace('.png', '', regex=False)
ids = set(pd.read_csv('splits/nih.csv').image_id)
m[m.image_id.isin(ids)].drop(columns=['Image Index']).to_csv('results/r1/nih_metadata.csv', index=False)
print('nih metadata rows', m.image_id.isin(ids).sum())
PY
fi
push "data ready"

# ---- weights + analyses
python scripts/fetch_weights.py --runs results/runs || echo "WARN some weights missing"
( while true; do sleep 1200; push "progress $(date -u +%H:%M)"; done ) &
python scripts/r1_gpu.py --data $DATA --runs results/runs --out results/r1 || echo "WARN r1_gpu exit $?"
echo done > results/r1/STATUS
stopnow "complete"
