# Inputs for the R1 revision analyses

- `siim_drain_labels.csv`: chest-drain labels for SIIM-ACR images matched (pHash Hamming <= 4) to NEATX-annotated
  NIH ChestX-ray14 images; produced in the companion repository (`ptx-drain-invariance`, `scripts/map_neatx.py`).
- `nih_neatx_drain.csv`: NEATX aggregated drain label for NIH device-cohort test images that NEATX annotated.

Source: NEATX non-expert drain annotations (Jiménez-Sánchez et al.), Zenodo 14944064, **CC BY-NC-SA 2.0**.
These derived files keep that licence (non-commercial, share-alike). Used for validation only; no model was trained on them.
