"""Figures for the manuscript, from data/analysis.json (seed-averaged NIH test results, 2,000 patient-cluster bootstraps)."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
D = json.load(open(Path(__file__).parent / "data" / "analysis.json"))
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

ORDER = [  # (run, label, group)
    ("pb_full", "Proposed (C2+C4)", "prop"),
    ("pb_no_c2", "Ablation: C4 only", "abl"),
    ("pb_no_c4", "Ablation: C2 only", "abl"),
    ("pb_base", "Ablation: neither (base)", "abl"),
    ("pb_sens_lambda001", r"Sensitivity: $\lambda$=0.01", "abl"),
    ("pb_sens_lambda1", r"Sensitivity: $\lambda$=1.0", "abl"),
    ("pb_bl_unet_r34_kaggle", "U-Net ResNet34", "bl"),
    ("pb_bl_segformer_b2", "SegFormer-B2", "bl"),
    ("pb_bl_groupdro", "Group DRO", "bl"),
    ("pb_bl_bandaru_auxcls", "Aux. device classifier", "bl"),
    ("pb_bl_raddino_probe", "RAD-DINO linear probe", "bl"),
    ("pb_bl_raddino_ft", "RAD-DINO fine-tuned", "bl"),
]


def style_y(ax, labels, groups):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for t, g in zip(ax.get_yticklabels(), groups):
        if g == "prop":
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for y, g in enumerate(groups):
        if g == "prop":
            ax.axhspan(y - 0.5, y + 0.5, color="#f0efec", zorder=0)
    ax.axhline(5.5, color=MUTED, lw=0.5, ls=":")


# ---------------- Figure 2: AUROC by device stratum, AUC gap, device leakage into score
runs, labels, groups = zip(*ORDER)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True, gridspec_kw={"width_ratios": [1.5, 1, 1]})
ax = axes[0]
for i, r in enumerate(runs):
    u, t = D["nih"][r]["auc_untr"], D["nih"][r]["auc_trt"]
    ax.plot([u[0], t[0]], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.errorbar(t[0], i - 0.13, xerr=[[t[0] - t[1]], [t[2] - t[0]]], fmt="o", ms=4, color=BLUE, lw=1, capsize=0, zorder=3)
    ax.errorbar(u[0], i + 0.13, xerr=[[u[0] - u[1]], [u[2] - u[0]]], fmt="s", ms=4, color=ORANGE, lw=1, capsize=0, zorder=3)
style_y(ax, labels, groups)
ax.set_xlim(0.55, 1.0)
ax.set_xlabel("AUROC vs all negatives\n(95% CI)")
ax.set_title("a  Detection by device stratum", loc="left")
ax.plot([], [], "o", color=BLUE, ms=4, label="Device-present positives")
ax.plot([], [], "s", color=ORANGE, ms=4, label="Untreated positives")

ax = axes[1]
for i, r in enumerate(runs):
    g = D["nih"][r]["gap"]
    ax.errorbar(g[0], i, xerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt="o", ms=4, color=INK if groups[i] == "prop" else MUTED, lw=1, capsize=0)
ax.axvline(0, color=MUTED, lw=0.6)
style_y(ax, labels, groups)
ax.set_xlabel("AUC gap\n(device $-$ untreated)")
ax.set_title("b  Device$-$untreated gap", loc="left")

ax = axes[2]
for i, r in enumerate(runs):
    g = D["nih"][r]["neg_dev_auc"]
    ax.errorbar(g[0], i, xerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt="o", ms=4, color=INK if groups[i] == "prop" else MUTED, lw=1, capsize=0)
ax.axvline(0.5, color=MUTED, lw=0.6)
style_y(ax, labels, groups)
ax.set_xlim(0.3, 0.8)
ax.set_xlabel("Device AUROC of the score\n(negatives only)")
ax.set_title("c  Device signal in the score", loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.2, 1.06), ncol=2, frameon=False, handletextpad=0.3)
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "fig2_shortcut.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3: paired differences, proposed minus comparator
comp = [o for o in ORDER if o[0] != "pb_full"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, key, title, xl in [
    (axes[0], "auc_untr", "a  Untreated AUROC: proposed $-$ comparator", "Difference (95% CI)"),
    (axes[1], "gap", "b  AUC gap: proposed $-$ comparator", "Difference (95% CI; >0 = larger gap)"),
]:
    for i, (r, lab, g) in enumerate(comp):
        v = D["paired"][f"{r}|{key}"]
        worse = (v[2] < 0) if key == "auc_untr" else (v[1] > 0)
        ax.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=4,
                    color=ORANGE if worse else MUTED, lw=1, capsize=0)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels([c[1] for c in comp])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(4.5, color=MUTED, lw=0.5, ls=":")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xl)
axes[0].plot([], [], "o", color=ORANGE, ms=4, label="CI excludes 0 (proposed worse)")
axes[0].plot([], [], "o", color=MUTED, ms=4, label="CI includes 0")
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.2, 1.07), ncol=2, frameon=False, handletextpad=0.3)
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "fig3_paired.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 4: per-seed untreated sensitivity at the transferred threshold (A1/A8)
ps = D["per_seed_sens_untr"]
order4 = [o for o in ORDER if o[0] in ps]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
rng = np.random.default_rng(0)
for i, (r, lab, g) in enumerate(order4):
    v = np.array(ps[r])
    x = i + rng.uniform(-0.12, 0.12, len(v))
    col = INK if g == "prop" else (BLUE if g == "abl" else MUTED)
    ax.scatter(x, v, s=18, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=col, lw=1.6, zorder=2)
ax.set_xticks(range(len(order4)))
SHORT = {"pb_full": "Proposed\n(C2+C4)", "pb_no_c2": "C4 only", "pb_no_c4": "C2 only", "pb_base": "Base", "pb_bl_unet_r34_kaggle": "U-Net\nRes34",
         "pb_bl_segformer_b2": "SegFormer\nB2", "pb_bl_groupdro": "Group\nDRO", "pb_bl_bandaru_auxcls": "Aux.\ndevice cls", "pb_bl_raddino_probe": "RAD-DINO\nprobe", "pb_bl_raddino_ft": "RAD-DINO\nfine-tuned"}
ax.set_xticklabels([SHORT[o[0]] for o in order4], fontsize=7)
for t, o in zip(ax.get_xticklabels(), order4):
    if o[2] == "prop":
        t.set_fontweight("bold")
ax.set_ylabel("Untreated sensitivity\n(NIH test, 36 positives)")
ax.set_ylim(0, 0.4)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.axvline(3.5, color=MUTED, lw=0.5, ls=":")
fig.tight_layout()
fig.savefig(OUT / "fig4_seeds.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
