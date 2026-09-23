# Pre-registration: Seeing the air, not the tube

Status: DRAFT v0.3 (2026-09-23, Amendment 1 added). Freeze with `scripts/freeze_prereg.sh` before the first external evaluation.
Any change after freezing is logged under *Deviations* with date and reason; the external-evaluation
script refuses to run if this file's hash differs from `results/prereg_hash.txt`.

## 1. Hypotheses

- **H1.** A pneumothorax model trained to ignore chest tubes (C1 tube inpainting, C2 device-adversarial
  loss, C3 auxiliary tube segmentation, C4 mask pooling) detects *untreated* pneumothorax better than
  current methods on external data.
- **H2.** The same model depends less on the tube: its performance gap between treated and
  untreated cases shrinks.

## 2. Data and roles

| Dataset | Role | Labels used | Notes |
| --- | --- | --- | --- |
| CANDID-PTX (19,237 CXRs) | Development: train / val / test_internal, 70/10/20 by patient | PTX masks, chest-tube masks | Needs ethics course + DUA |
| SIIM-ACR (Kaggle, ~10.7k with labels) | Primary external 1 | PTX masks; tube status by pseudo-label (tube segmenter trained on CANDID train only) + manual check of every positive | |
| NIH device cohort (Zenodo 22839698) | Primary external 2 (released `test` split, 878 images) | PTX vs no finding; *any visible device* | Device is not tube-specific; its `train`/`validation` splits are used only for the Gate-1 pilot |
| PTX-498 (Zenodo 8266529, v2-fix) | Secondary external | PTX masks, 3 sites | **Positives only**: used for sensitivity, Dice and size; not for AUROC or specificity |

"Untreated" = pneumothorax present and `treated == 0` (no chest tube; for NIH, no visible device).
Near-duplicates (pHash Hamming <= 4) of any development image are removed from external sets.
Splits are generated once (`ptx.data.splits`, seed 2026) and committed.

## 3. Operating point

Each model's threshold is the 95%-specificity point on **development validation**, computed on
seed-averaged scores, and applied unchanged to all other sets.

## 4. Endpoints and pass thresholds

"Best baseline" = the strongest re-run baseline for that dataset and that metric
(highest sensitivity or AUROC; smallest gap for A2). Seed-averaged predictions; 95% CIs by
patient-cluster bootstrap (2,000 resamples).

| ID | Metric | Pass | Where |
| --- | --- | --- | --- |
| A1 (primary) | Sensitivity on untreated positives at the fixed threshold | Difference vs best baseline >= +0.05 and CI lower bound > 0 | SIIM **and** NIH |
| A2 (primary) | AUC gap = AUROC(treated vs negatives) - AUROC(untreated vs negatives) | Reduced by >= 50% vs the smallest-gap baseline | SIIM **and** NIH |
| A3 | AUROC, untreated positives vs all negatives | >= +0.02 vs best baseline, paired DeLong p < 0.05 after Holm across baselines | >= 1 primary external |
| A4 | Sensitivity on small PTX (reference rim < 20 mm; else lowest tertile of reference area fraction) | Not below best baseline (>= +0.03 = win) | >= 1 set with masks |
| A5 | Dice on positives | Non-inferior: 95% one-sided lower bound of difference > -0.01 vs best segmentation baseline | CANDID test_internal and PTX-498 |
| A6 | Weighted kappa (linear), predicted vs reference size class (BTS rim >= 20 mm when spacing known; else area tertiles) | >= 0.70 | CANDID test_internal and PTX-498 |
| A7 | ECE (15 bins) after temperature scaling fitted on dev validation | <= 0.05 | SIIM and NIH |
| A8 | Seed stability of the A1 metric | SD <= 0.02 over 5 seeds and >= 4/5 seeds above the best baseline's seed mean | SIIM and NIH |

**Accepted as positive** = A1 and A2 pass, at least 4 of A3-A8 pass, and all sanity checks pass.

## 5. Sanity checks

- **Occlusion:** images reduced to the dilated tube region only. The proposed model's positive rate at
  its threshold must be below every baseline's, and its AUROC for the PTX label on these images <= 0.60.
- **Device probe:** 5-fold logistic regression predicting `treated` from pooled bottleneck features.
  Proposed-model AUROC must be below every baseline's.
- **Leakage:** zero patient overlap across development splits and no dev/test near-duplicates.

## 6. Ablations

Runs: full, -C1, -C2, -C3, -C4, Base (no C1-C4), 5 seeds each. Component k is **positive** if removing
it lowers the A1 metric (mean over seeds) on both primary externals and the drop appears in >= 4 of 5
paired seeds; for Base, the CI of the full-vs-Base difference must also exclude 0 on both sets.
Components that fail are removed from the final model and reported as not helping.
Sensitivity runs (3 seeds): adversarial weight 0.01 and 1.0; mean-fill instead of Telea inpainting.

## 7. Baselines (all re-run on the same splits, 512 px, same augmentation)

U-Net ResNet34 (SIIM 1st-place family), U-Net EfficientNet-B4, SegFormer-B2, nnU-Net 2D, Group DRO
(groups = PTX x treated), auxiliary tube classifier (Bandaru et al. 2026), tube-as-extra-class
segmentation (after Rueckel et al. 2021), RAD-DINO linear probe and fine-tuned, zero-shot MLLMs.
Prior published comparator on the NIH cohort (Zamora Zamora et al., Zenodo 22839698): their mitigation
reduced the device sensitivity gap by 29% (CI excludes 0) with AUROC 0.86 vs 0.87 conventional.

## 8. Statistics

Seeds 0-4 (3 for heavy baselines). Mean +- SD across seeds reported for every metric; every seed
reported in the supplement. Holm correction within each dataset across baselines and across ablations.
No seed is dropped after its result is seen. External sets are evaluated once per frozen model.

## 9. Gates

- **Gate 1 (week 3):** on development validation only, some deconfounding variant must gain >= 2 pp
  untreated sensitivity at 95% specificity over the baseline; otherwise switch to the shortcut-audit paper.
- **Gate 2 (week 6):** freeze models, open externals once, run `ptx.eval.acceptance`.

## Amendment 1 — Plan B without CANDID-PTX (2026-09-23, before any external evaluation)

Access to CANDID-PTX is pending (ethics course + data-use agreement). To proceed, a second protocol
runs in parallel and is reported as such. If CANDID-PTX arrives before Gate 2, the original protocol
(sections 1-9) is the primary analysis and Plan B becomes a sensitivity analysis.

- **Development set:** SIIM-ACR, 70/10/20 split (`ptx.data.splits --dev siim`, seed 2026).
  SIIM-ACR has one identifier per image, so splits are image-level plus near-duplicate grouping.
- **"Treated" label:** any visible device, predicted by an image classifier trained only on the NIH
  cohort's released `train` split and thresholded on its `validation` split (`ptx.data.device_labels`).
  Every SIIM positive is listed for manual review. The NIH `test` split is never used for this.
- **External sets:** NIH cohort `test` (real device labels) is the single primary external set;
  PTX-498 is secondary (sensitivity, Dice, size). SIIM-ACR images derive from NIH ChestX-ray14, so
  near-duplicates of SIIM images are removed from the NIH test split (reported count).
- **Method:** C2 (device-adversarial) + C4 (mask pooling). C1 and C3 need tube masks and are dropped.
- **Criteria changes:** A1 and A2 must pass on the one primary external set (min_primary_sets = 1);
  A7 and A8 on the NIH test set; A5 and A6 on SIIM test_internal and PTX-498. Sanity checks: device probe
  and leakage (the tube-only occlusion test needs tube masks and is dropped). Ablations: -C2, -C4,
  Base; the ablation rule applies to the single primary external set.
- **Limitation stated in advance:** "device" is not "chest tube"; results speak to device shortcuts in
  general, and patient-level overlap between SIIM and NIH cannot be fully excluded (only image-level).

## Deviations

(none yet)
