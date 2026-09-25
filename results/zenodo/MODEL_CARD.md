# Model card: ptx-deconfound checkpoints

**Task.** Image-level pneumothorax detection (and, for segmentation models, pixel masks) on frontal chest radiographs.
**Training data.** SIIM-ACR (development split in the code repository); device labels on SIIM-ACR are pseudo-labels.
**Evaluation.** NIH ChestX-ray14-derived device cohort (primary external) and PTX-498 (secondary); see the paper.
**Key limitation.** Every configuration relies partly on visible devices (a device shortcut): performance on
untreated pneumothorax is markedly lower than on device-present cases, and decision thresholds do not
transfer between datasets without recalibration. Research use only; not for clinical decisions.

**Contents.** 54 checkpoints from 12 configurations: pb_base, pb_bl_bandaru_auxcls, pb_bl_groupdro, pb_bl_raddino_ft, pb_bl_raddino_probe, pb_bl_segformer_b2, pb_bl_unet_r34_kaggle, pb_full, pb_no_c2, pb_no_c4, pb_sens_lambda001, pb_sens_lambda1.
File name = `<configuration>__seed<k>.pt`; architecture and hyper-parameters are in
`results/runs/<configuration>/seed<k>/config.yaml` of https://github.com/yazanjer/ptx-deconfound.

**Licence.** CC-BY-4.0 for the weights; the code is MIT. Dataset licences of SIIM-ACR and NIH ChestX-ray14 apply.
