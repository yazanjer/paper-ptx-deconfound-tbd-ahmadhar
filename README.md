# ptx-deconfound

Code, frozen splits, per-image predictions and manuscript sources for the study

**"Adversarial deconfounding with pseudo device labels does not remove the device shortcut in pneumothorax
detection: a pre-registered, multi-seed external evaluation of twelve models."**

Status: complete. 54 training runs (12 configurations, 3–5 seeds) were trained on SIIM-ACR and evaluated
once on the NIH ChestX-ray14 device cohort (primary external) and PTX-498 (secondary), after the protocol in
[PREREGISTRATION.md](PREREGISTRATION.md) was frozen (SHA-256 recorded in the repository). The proposed
model did not outperform the baselines; every model showed a device shortcut. See `paper/` and
`results/planb/`.

## What's here

| Path | Contents |
| --- | --- |
| `ptx/data/` | Dataset preparation (SIIM-ACR, NIH device cohort, PTX-498, CANDID-PTX), frozen splits, near-duplicate removal, device pseudo-labels, leakage check |
| `ptx/models/net.py` | U-Net-family model with C2 (gradient-reversal device adversary) and C4 (top-k mask pooling); RAD-DINO probe |
| `ptx/train.py` | One run = one config x one seed; picks the 95%-specificity threshold on development validation |
| `ptx/evaluate_external.py` | Opens the external test sets once, guarded by the pre-registration hash |
| `ptx/eval/` | Metrics, size estimation, DeLong, patient-cluster bootstrap, Holm, evaluation report |
| `configs/` | `base_planb.yaml`, one file per run in `runs/`, `experiments.yaml` (stage queue) |
| `scripts/` | RunPod bootstrap, queue runner, result push, weight upload/fetch, Zenodo upload, paper-figure rendering |
| `splits/` | Frozen split files |
| `results/runs/` | Per-run config, training log, per-epoch validation metrics, per-image predictions |
| `results/planb/` | External evaluation report, tables, leakage check |
| `results/paper_figs/` | Example radiographs and seed-averaged model maps used in the paper |
| `paper/` | Manuscript (LaTeX, Elsevier `elsarticle`), bibliography, figure scripts and figure data |

## Model weights

- GitHub release `weights-v1`: 54 checkpoints (`<run>__seed<k>.pt`).
- Zenodo: the same checkpoints with `manifest.csv` (SHA-256, MD5, thresholds) and `MODEL_CARD.md`.
  They are uploaded by `scripts/upload_zenodo.py` (stage `zenodo_models`) as a draft, and published together
  with the code when the repository becomes public.

```python
import torch, yaml
from ptx.models.net import build_model
cfg = yaml.safe_load(open("results/runs/pb_full/seed0/config.yaml"))
m = build_model(cfg); m.load_state_dict(torch.load("pb_full__seed0.pt", map_location="cpu")); m.eval()
```

## Pipeline (Plan B)

```
data -> device pseudo-labels (planb_prep) -> pilot (planb_gate1) -> 54 runs (planb_main_a/b/b2/c)
     -> freeze pre-registration -> single external evaluation (planb_gate2) -> paper_figs -> zenodo_models
```

## Running on RunPod

1. In RunPod -> Secrets, create `github_token` (fine-grained PAT, this repo only, Contents read/write),
   Kaggle credentials for data download, and `zenodo` (Zenodo personal token, scopes deposit:write and
   deposit:actions) for the model upload.
2. Deploy a pod from the **ptx-deconfound** template: 1 GPU, 150 GB volume at `/workspace`, environment
   `STAGES` (e.g. `data planb_prep planb_main_a`, `planb_gate2`, `zenodo_models`), `MAX_HOURS`, and the
   secrets as `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `ZENODO_TOKEN`.
3. The pod clones the repo, runs the stages, pushes results after every step, and stops itself when done.

Local equivalent:

```bash
pip install -r requirements.txt
python scripts/run_queue.py --stage planb_main_a --data data
python scripts/fetch_weights.py --runs results/runs        # needs GITHUB_TOKEN
```

## Paper

`paper/main.tex` builds with `pdflatex` + `bibtex`: first run `latex elsarticle.ins` in `paper/tex/` and copy
`elsarticle.cls` and `elsarticle-num-names.bst` next to `main.tex`. Figures: `python paper/make_figures.py`, `make_figures2.py`, `make_figures3.py`.

## Data access

Images are not redistributed. SIIM-ACR and NIH ChestX-ray14 via Kaggle; PTX-498 and the NIH device cohort
manifest via Zenodo.

## Tests

`python -m pytest -q` runs unit tests for the statistics and an end-to-end run on synthetic data with a
planted device shortcut (about 2 minutes on CPU).

## License

Code: MIT. Results files and model weights: CC-BY 4.0. Dataset licences apply to the underlying images.
