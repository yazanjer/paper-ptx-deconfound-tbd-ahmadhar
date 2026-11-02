#!/usr/bin/env bash
# R1: fetch ChestX-ray14 metadata (Data_Entry_2017.csv) and the NIH device-cohort manifest on a CPU pod,
# keep only the cohort images, push to branch r1-nihmeta, stop. No images, no GPU.
set -uo pipefail
export GIT_TERMINAL_PROMPT=0
GH_REPO=${GH_REPO:-yazanjer/paper-ptx-deconfound-tbd-ahmadhar}
W=/w; mkdir -p $W && cd $W
exec > >(tee -a $W/log.txt) 2>&1
echo "=== meta start $(date -u)"
( sleep 2700; runpodctl stop pod "$RUNPOD_POD_ID" ) &
(apt-get update -qq && apt-get install -y -qq git curl > /dev/null) || true
pip install -q kaggle pandas 2>&1 | tail -1
git clone -q --depth 1 -b r1-ieee "https://x-access-token:${GITHUB_TOKEN}@github.com/${GH_REPO}.git" repo || { echo FATAL clone; sleep 300; runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity; }
mkdir -p ~/.kaggle && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
for i in 1 2 3; do kaggle datasets download -d nih-chest-xrays/data -f Data_Entry_2017.csv -p $W && break; sleep 30; done
ls -la $W
for f in $W/Data_Entry_2017.csv.zip $W/Data_Entry_2017.csv*.zip; do [ -f "$f" ] && python -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall('$W')" "$f"; done
for i in 1 2 3; do curl -sL -o $W/cohort.zip "https://zenodo.org/api/records/22839698/files/Pneumothorax_Zenodo_READY_v1.0.0.zip/content" && python -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall('$W/cohort')" $W/cohort.zip && break; sleep 30; done
cd repo && mkdir -p results/r1 && python - <<'PY'
import glob, pandas as pd
ids = set(pd.read_csv('splits/nih.csv').image_id)
try:
    m = pd.read_csv('/w/Data_Entry_2017.csv'); m['image_id'] = m['Image Index'].str.replace('.png', '', regex=False)
    m[m.image_id.isin(ids)].drop(columns=['Image Index']).to_csv('results/r1/nih_metadata.csv', index=False)
    print('data entry rows', m.image_id.isin(ids).sum())
except Exception as e:
    print('no Data_Entry', e)
c = glob.glob('/w/cohort/**/cohort_manifest.csv', recursive=True)
if c:
    k = pd.read_csv(c[0]); print('cohort columns', list(k.columns))
    k['image_id'] = k.image_id.astype(str).str.replace('.png', '', regex=False)
    k[k.image_id.isin(ids)].to_csv('results/r1/nih_cohort_manifest.csv', index=False)
PY
cp $W/log.txt results/r1/meta_log.txt
git -c user.name=ptx-runner -c user.email=ptx-runner@users.noreply.github.com add -f results/r1 && git -c user.name=ptx-runner -c user.email=ptx-runner@users.noreply.github.com commit -qm "r1: NIH metadata" && git push -qf origin HEAD:refs/heads/r1-nihmeta && echo pushed
sleep 30; runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity
