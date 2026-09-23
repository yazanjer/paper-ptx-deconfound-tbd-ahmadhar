# ptx-deconfound

**Seeing the air, not the tube.** A device-stratified, size-aware benchmark and a deconfounded model
for detecting *untreated* pneumothorax on chest radiographs.

Status: code complete and tested on synthetic data; waiting for data access and GPU runs.
Endpoints are fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

## What's here

| Path | Contents |
| --- | --- |
| `ptx/data/` | Dataset preparation (SIIM-ACR, CANDID-PTX, NIH device cohort, PTX-498), frozen splits, near-duplicate removal, chest-tube pseudo-labels, leakage check |
| `ptx/models/net.py` | U-Net-family model with C2 (device-adversarial), C3 (aux tube segmentation), C4 (mask pooling); RAD-DINO probe |
| `ptx/data/dataset.py` | C1 tube inpainting (with decoy inpainting on tube-free images) |
| `ptx/train.py` | One run = one config x one seed; picks the 95%-specificity threshold on dev validation |
| `ptx/evaluate_external.py` | Opens external test sets once (Gate 2), guarded by the pre-registration hash |
| `ptx/eval/` | Metrics, size estimation, DeLong, cluster bootstrap, Holm, acceptance checker (A1-A8, sanity, ablations) |
| `configs/` | `base.yaml`, one file per run in `runs/`, `experiments.yaml` (queue), `acceptance.yaml` |
| `scripts/` | RunPod bootstrap, queue runner, result push, weight upload, nnU-Net and MLLM baselines |
| `splits/` | Frozen split files (committed once created) |
| `results/` | Per-image predictions, logs, tables and the acceptance report (weights are release assets) |

## Pipeline

```
data -> splits (frozen) -> Gate 1 pilot -> tube pseudo-labels (SIIM) -> main runs (5 seeds)
     -> freeze pre-registration -> external evaluation (once) -> acceptance report
```

## Running on RunPod

1. In RunPod -> Secrets, create `github_token` (fine-grained PAT, only this repo, Contents read/write),
   `kaggle_username` and `kaggle_key`.
2. Deploy a pod from the **ptx-deconfound** template (id `nureplx93z`): 1x RTX 4090 or A40,
   150 GB volume at `/workspace`. Edit `STAGES` per step: `data gate1_nih`, then `data gate1`
   (after adding `CANDID_URL`), then `tube_labels main`, then `gate2`.
3. The pod clones the repo, prepares data, runs the queue, pushes results after every run,
   uploads weights as release assets and **stops itself** when done (billing stops; the volume stays).

Local equivalent:

```bash
pip install -r requirements.txt
python -m ptx.data.prepare ptx498 --raw raw/ptx498 --out data
python -m ptx.data.splits --data data --dev candid --out splits
python scripts/run_queue.py --stage gate1 --data data
```

## Data access

Images are not redistributed. SIIM-ACR and NIH ChestX-ray14 via Kaggle; PTX-498 and the NIH device
cohort manifest via Zenodo; CANDID-PTX via the University of Auckland after an ethics course and a
data-use agreement.

## Tests

`python -m pytest -q` runs unit tests for the statistics and an end-to-end run on synthetic data with a
planted chest-tube shortcut (about 2 minutes on CPU).

## License

Code: MIT. Results files: CC-BY 4.0. Dataset licences apply to the underlying images.
