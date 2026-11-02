"""Convergence curves (mean with min-max band over seeds) and a model x metric heatmap."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
H = Path(__file__).parent
C = json.load(open(H / "data" / "convergence.json"))
A = json.load(open(H / "data" / "analysis.json"))["nih"]
OUT = H / "figures"
GRID, MUTED, INK = "#e4e3df", "#52514e", "#0b0b0b"

FAM = [("pb_full", "Proposed (C2+C4)", INK, "-"), ("pb_no_c2", "C4 only", "#2a78d6", "-"),
       ("pb_no_c4", "C2 only", "#eb6834", "-"), ("pb_base", "Base", "#1baf7a", "-"),
       ("pb_sens_lambda001", r"C2, $\lambda$=0.01", "#8e5bd0", "--"), ("pb_sens_lambda1", r"C2, $\lambda$=1.0", "#b8860b", "--")]
BL = [("pb_bl_unet_r34_kaggle", "U-Net ResNet34", "#2a78d6", "-"), ("pb_bl_segformer_b2", "SegFormer-B2", "#eb6834", "-"),
      ("pb_bl_groupdro", "Group DRO", "#1baf7a", "-"), ("pb_bl_bandaru_auxcls", "Aux. device cls", "#8e5bd0", "-"),
      ("pb_bl_raddino_probe", "RAD-DINO probe", "#b8860b", "--"), ("pb_bl_raddino_ft", "RAD-DINO fine-tuned", INK, "--")]
METRICS = [("cls", "Train cls. loss"), ("val_auroc", "Validation AUROC"),
           ("val_sens95_untreated", "Val. device-absent sens."), ("val_dice_pos", "Val. Dice (positives)")]


def band(ax, run, key, col, ls, lab=None):
    if key not in C[run]:
        return
    m = C[run][key]
    x = np.arange(1, len(m["mean"]) + 1)
    ax.fill_between(x, m["min"], m["max"], color=col, alpha=0.13, lw=0)
    ax.plot(x, m["mean"], color=col, ls=ls, lw=1.1 if run != "pb_full" else 1.6, label=lab)


fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.0), sharex=True)
for r, group in enumerate([FAM, BL]):
    for c, (key, title) in enumerate(METRICS):
        ax = axes[r, c]
        for run, lab, col, ls in group:
            band(ax, run, key, col, ls, lab if c == 0 else None)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(title, loc="left")
        if r == 1:
            ax.set_xlabel("Epoch")
        if key == "cls":
            ax.set_yscale("log")
    axes[r, 0].set_ylabel(["Proposed model and variants", "Baselines"][r])
for ax in axes.flat:
    ax.set_xticks([1, 5, 10, 15, 20])
fig.tight_layout(h_pad=0.8, w_pad=0.6, rect=(0, 0.1, 1, 1))
for r in range(2):
    h, l = axes[r, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.02 + 0.5 * r, -0.005), ncol=2, frameon=False, fontsize=6.8,
               title=["Proposed model and variants", "Baselines"][r], title_fontsize=7, handlelength=1.8, columnspacing=1)
fig.savefig(OUT / "fig_convergence.pdf", bbox_inches="tight")
plt.close(fig)

# auxiliary losses: segmentation loss and adversary / auxiliary device loss
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for run, lab, col, ls in FAM + [(r_, l_, c_, ":") for r_, l_, c_, _ in BL[:4]]:
    band(axes[0], run, "seg", col, ls, lab)
for run, lab, col, ls in FAM:
    band(axes[1], run, "adv", col, ls, lab + " (adversary)" if "adv" in C[run] else None)
band(axes[1], "pb_bl_bandaru_auxcls", "aux_cls", MUTED, ":", "Aux. device cls (no reversal)")
p0 = 0.5 * 0.869 + 0.5 * 0.553  # device prevalence in batches with 50% positives
H0 = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
axes[1].axhline(H0, color=MUTED, lw=0.6, ls="-.")
axes[1].text(20, H0 + 0.03, "chance level (label entropy)", ha="right", va="bottom", fontsize=6.5, color=MUTED)
axes[0].set_title("a  Segmentation loss (BCE + soft Dice)", loc="left")
axes[1].set_title("b  Device-head loss", loc="left")
for ax in axes:
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 5, 10, 15, 20])
axes[0].legend(frameon=False, ncol=3, fontsize=6.3, handlelength=1.5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
axes[1].legend(frameon=False, fontsize=6.3, handlelength=1.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25))
axes[1].set_ylim(0, 1.2)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig_aux_losses.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- heatmap: models x metrics, coloured by within-column rank (better = darker)
SEC = {  # SIIM AUROC, SIIM Dice, PTX-498 sens, PTX-498 Dice (Table secondary)
    "pb_full": (0.957, 0.460, 0.693, 0.478), "pb_no_c2": (0.957, 0.453, 0.734, 0.496), "pb_no_c4": (0.950, 0.508, 0.714, 0.730),
    "pb_base": (0.955, 0.477, 0.660, 0.643), "pb_bl_unet_r34_kaggle": (0.954, 0.453, 0.622, 0.485),
    "pb_bl_segformer_b2": (0.959, 0.499, 0.830, 0.731), "pb_bl_groupdro": (0.952, 0.458, 0.782, 0.595),
    "pb_bl_bandaru_auxcls": (0.956, 0.523, 0.643, 0.699), "pb_bl_raddino_probe": (0.915, np.nan, 0.382, np.nan),
    "pb_bl_raddino_ft": (0.949, np.nan, 0.635, np.nan)}
ROWS = [(r, l) for r, l, *_ in FAM] + [(r, l) for r, l, *_ in BL]
COLS = [("NIH\nAUROC", 1), ("NIH dev.-abs.\nAUROC", 1), ("NIH device\nAUROC", 0), ("Gap", -1), ("Device-in-\nscore", -1),
        ("Oracle sens.\ndev.-abs.", 1), ("SIIM\nAUROC", 1), ("SIIM\nDice", 1), ("PTX-498\nsens.", 1), ("PTX-498\nDice", 1)]
M = np.full((len(ROWS), len(COLS)), np.nan)
for i, (r, _) in enumerate(ROWS):
    a = A[r]
    M[i, :6] = [a["auroc"][0], a["auc_untr"][0], a["auc_trt"][0], a["gap"][0], a["neg_dev_auc"][0], a["oracle_sens95_untr"]]
    if r in SEC:
        M[i, 6:] = SEC[r]
S = np.full_like(M, np.nan)
for j, (_, sgn) in enumerate(COLS):
    col = M[:, j]
    ok = ~np.isnan(col)
    lo, hi = np.nanmin(col), np.nanmax(col)
    if sgn == 0:
        S[ok, j] = 0.0
        continue
    S[ok, j] = ((col[ok] - lo) / (hi - lo + 1e-12)) if sgn > 0 else ((hi - col[ok]) / (hi - lo + 1e-12))
cmap = LinearSegmentedColormap.from_list("q", ["#ffffff", "#d6e6f7", "#7fb0e3", "#2a78d6", "#123f78"])
cmap.set_bad("#f3f3f1")
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.imshow(np.ma.masked_invalid(S), cmap=cmap, aspect="auto", vmin=0, vmax=1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=6.6,
                color="white" if (not np.isnan(S[i, j]) and S[i, j] > 0.72) else INK)
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([c for c, _ in COLS], fontsize=6.8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([l for _, l in ROWS])
ax.get_yticklabels()[0].set_fontweight("bold")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.axhline(5.5, color="white", lw=2.5)
for x in (5.5,):
    ax.axvline(x, color="white", lw=2.5)
ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
ax.grid(which="minor", color="white", lw=0.8)
ax.tick_params(which="minor", length=0)
fig.tight_layout()
fig.savefig(OUT / "fig_heatmap.pdf", bbox_inches="tight")
plt.close(fig)
print("ok")
