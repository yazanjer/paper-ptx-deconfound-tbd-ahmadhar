"""Figures added in revision (R1). Inputs: results/r1/cpu.json (scripts/r1_cpu.py), results/r1/gpu_summary.json
(scripts/r1_summarize_gpu.py) and the stored per-image predictions. Writes PDF (vector) and 300-dpi PNG."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptx.eval.acceptance import load_run  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
H = Path(__file__).resolve().parent
ROOT = H.parent
J = json.load(open(ROOT / "results/r1/cpu.json"))
OUT = H / "figures"
ORDER = [
    ("pb_full", "Proposed (C2+C4)", "prop"), ("pb_no_c2", "Ablation: C4 only", "abl"), ("pb_no_c4", "Ablation: C2 only", "abl"),
    ("pb_base", "Ablation: neither (base)", "abl"), ("pb_sens_lambda001", r"Sensitivity: $\lambda$=0.01", "abl"),
    ("pb_sens_lambda1", r"Sensitivity: $\lambda$=1.0", "abl"), ("pb_bl_unet_r34_kaggle", "U-Net ResNet34", "bl"),
    ("pb_bl_segformer_b2", "SegFormer-B2", "bl"), ("pb_bl_groupdro", "Group DRO", "bl"),
    ("pb_bl_bandaru_auxcls", "Aux. device classifier", "bl"), ("pb_bl_raddino_probe", "RAD-DINO linear probe", "bl"),
    ("pb_bl_raddino_ft", "RAD-DINO fine-tuned", "bl")]


def style_y(ax, labels, groups):
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for t, g in zip(ax.get_yticklabels(), groups):
        if g == "prop":
            t.set_fontweight("bold")
    ax.invert_yaxis(); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for y, g in enumerate(groups):
        if g == "prop":
            ax.axhspan(y - 0.5, y + 0.5, color="#f0efec", zorder=0)
    ax.axhline(5.5, color=MUTED, lw=0.5, ls=":")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


runs, labels, groups = zip(*ORDER)

# ---------------- Figure R1: positive strata on NIH (drain / other device / no device) and size-stratified gap on SIIM-ACR
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
D, R = J["drain_strata"], J["nih_reproduced"]
for i, r in enumerate(runs):
    v = [D[r]["nih_auc_pos_drain"], D[r]["nih_auc_pos_device_nodrain"], R[r]["auroc_untr"]]
    ax.plot([min(v), max(v)], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.plot(v[0], i, "o", color=BLUE, ms=4, zorder=3)
    ax.plot(v[1], i, "D", color=AQUA, ms=3.6, zorder=3)
    ax.plot(v[2], i, "s", color=ORANGE, ms=4, zorder=3)
style_y(ax, labels, groups)
ax.plot([], [], "o", color=BLUE, ms=4, label="Chest drain (NEATX), n=42")
ax.plot([], [], "D", color=AQUA, ms=3.6, label="Other device, no drain, n=18")
ax.plot([], [], "s", color=ORANGE, ms=4, label="No visible device, n=36")
ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0, 1.0), ncol=1, handletextpad=0.3)
ax.set_xlabel("AUROC vs all NIH negatives (n=396)")
ax.set_title("a  NIH test: positives by device type", loc="left", pad=46)
ax.set_xlim(0.6, 1.005)
ax = axes[1]
S = J["severity_siim_test"]
mk = {"t0": ("v", "#9cc3ef"), "t1": ("o", BLUE), "t2": ("^", "#16467f")}
for i, r in enumerate(runs):
    if r not in S:
        continue
    for k, (m, c) in mk.items():
        ax.plot(S[r][k]["gap"], i + {"t0": -0.2, "t1": 0, "t2": 0.2}[k], m, color=c, ms=3.6, zorder=3)
    ax.plot(S[r]["gap_unadjusted"], i, "|", color=INK, ms=9, mew=1.2, zorder=4)
ax.axvline(0, color=MUTED, lw=0.6)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0); ax.axhline(5.5, color=MUTED, lw=0.5, ls=":")
for k, lab in [("t0", "Smallest size tertile"), ("t1", "Middle tertile"), ("t2", "Largest tertile")]:
    ax.plot([], [], mk[k][0], color=mk[k][1], ms=3.6, label=lab)
ax.plot([], [], "|", color=INK, ms=9, mew=1.2, label="All sizes")
ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
ax.set_xlabel("Device $-$ device-absent AUROC\n(SIIM-ACR internal test, pseudo-labels)")
ax.set_title("b  Gap within reference-mask size tertiles", loc="left", pad=46)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_r1_strata")

# ---------------- Figure R1: threshold transfer (score distributions)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
bins = np.linspace(-9, 9, 46)
for ax, r, ttl in [(axes[0], "pb_full", "a  Proposed: negatives"), (axes[1], "pb_bl_raddino_ft", "b  RAD-DINO fine-tuned: negatives")]:
    L = load_run(ROOT / "results/runs", r, "siim", "nih", "test")
    v, n = L["val"], L["df"]
    ax.hist(lg(v[v.ptx == 0].score), bins, density=True, histtype="stepfilled", color=GREY, alpha=0.45, label="SIIM-ACR validation")
    ax.hist(lg(n[(n.ptx == 0) & (n.treated == 1)].score), bins, density=True, histtype="step", color=BLUE, lw=1.1, label="NIH, device present")
    ax.hist(lg(n[(n.ptx == 0) & (n.treated == 0)].score), bins, density=True, histtype="step", color=ORANGE, lw=1.1, label="NIH, device absent")
    ax.axvline(lg(L["thr"]), color=INK, lw=0.8, ls="--")
    ax.set_title(ttl, loc="left"); ax.set_xlabel("Seed-averaged score (logit scale)"); ax.set_yticks([])
axes[0].legend(frameon=False, loc="upper right", fontsize=6.5)
ax = axes[2]
T = J["threshold_transfer"]
x = np.arange(len(runs))
ax.scatter([T[r]["neg_q95_siim_val"] for r in runs], [T[r]["neg_q95_nih"] for r in runs], s=14, color=BLUE, zorder=3)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("SIIM-ACR validation"); ax.set_ylabel("NIH test")
ax.set_title("c  95th percentile of negative scores", loc="left")
fig.tight_layout(w_pad=0.8)
save(fig, "fig_r1_threshold")

# ---------------- Figure R1: top-k overlap (if the GPU summary exists)
G = ROOT / "results/r1/gpu_summary.json"
if G.exists():
    S = json.load(open(G))["topk"]
    seg_runs = [o for o in ORDER if o[0] in S]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, ds, ttl in [(axes[0], "siim", "a  SIIM-ACR internal test positives"), (axes[1], "ptx498", "b  PTX-498 (three sites)")]:
        for i, (r, _, _) in enumerate(seg_runs):
            s = S[r].get(ds)
            if not s:
                continue
            ax.plot(s["topk_in_gt"], i, "o", color=BLUE, ms=4, zorder=3)
            ax.plot(s["topk_in_gt_dil"], i, "s", color=AQUA, ms=3.6, zorder=3)
            ax.plot(s["argmax_in_gt"], i, "^", color=ORANGE, ms=4, zorder=3)
            ax.plot(s["chance_in_gt"], i, "|", color=MUTED, ms=8, mew=1, zorder=2)
        if ax is axes[0]:
            style_y(ax, [o[1] for o in seg_runs], [o[2] for o in seg_runs])
        else:
            ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlim(0, 1); ax.set_title(ttl, loc="left"); ax.set_xlabel("Fraction")
    axes[0].plot([], [], "o", color=BLUE, ms=4, label="Top-$k$ pixels inside mask")
    axes[0].plot([], [], "s", color=AQUA, ms=3.6, label="Top-$k$ pixels within 25 px of mask")
    axes[0].plot([], [], "^", color=ORANGE, ms=4, label="Maximum pixel inside mask")
    axes[0].plot([], [], "|", color=MUTED, ms=8, mew=1, label="Mask area (chance)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_r1_topk")
print("ok")
