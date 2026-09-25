"""Dataset example grid and seed-averaged model maps from results/paper_figs (rendered on GPU by scripts/render_paper_figs.py)."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
H = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p for p in [H / "data" / "paper_figs", H.parent / "results" / "paper_figs"] if (p / "index.json").exists())
OUT = H / "figures"
IDX = {r["name"]: r for r in json.load(open(SRC / "index.json"))}


def load(p, size=None):
    im = Image.open(p).convert("L")
    if size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def outline(ax, m, color="#e8412c"):
    if m is not None and m.max() > 0:
        ax.contour(m > 0.5, levels=[0.5], colors=[color], linewidths=0.7)


DEV = {0: "no device", 1: "device"}
TITLES = {"untreated_tp": "NIH untreated PTX\n(highest score)", "untreated_fn": "NIH untreated PTX\n(lowest score)",
          "device_pos": "NIH PTX with device\n(median score)", "device_neg_fp": "NIH no PTX, device\n(highest score)",
          "clean_neg": "NIH no PTX, no device\n(median score)"}

# ---------------- Figure: dataset examples
names = [n for n in ["untreated_tp", "untreated_fn", "device_pos", "device_neg_fp", "clean_neg"] if n in IDX]
names += sorted(n for n in IDX if n.startswith("siim_")) + sorted(n for n in IDX if n.startswith("ptx498_"))
ncol = 5
nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.62 * nrow))
for ax in axes.flat:
    ax.axis("off")
for ax, n in zip(axes.flat, names):
    r = IDX[n]
    img = load(SRC / n / "img.png", (512, 512))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    mp = SRC / n / "mask.png"
    outline(ax, load(mp, (512, 512)) if mp.exists() else None)
    if r["dataset"] == "nih":
        t = TITLES[n]
    elif r["dataset"] == "siim":
        t = f"SIIM-ACR test PTX\n(pseudo-label: {DEV.get(r['treated'], '?')})"
    else:
        t = f"PTX-498 {n.split('_', 1)[1].replace('Site', 'site ')}\n(PTX, mask)"
    ax.set_title(t, fontsize=6.6, pad=2)
fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.4)
fig.savefig(OUT / "fig_examples.pdf", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---------------- Figure: seed-averaged maps on the four NIH cases
CASES = [n for n in ["untreated_tp", "untreated_fn", "device_pos", "device_neg_fp"] if n in IDX]
MODELS = [("pb_full", "Proposed\n(C2+C4)"), ("pb_no_c2", "C4 only"), ("pb_base", "Base"),
          ("pb_bl_segformer_b2", "SegFormer-B2"), ("pb_bl_groupdro", "Group DRO")]
MODELS = [m for m in MODELS if (SRC / CASES[0] / f"seg_{m[0]}.png").exists()]
nr, nc = 1 + len(MODELS), 2 * len(CASES)
fig, axes = plt.subplots(nr, nc, figsize=(7.2, 0.93 * nr + 0.35), gridspec_kw={"wspace": 0.03, "hspace": 0.06})
for j, n in enumerate(CASES):
    img = load(SRC / n / "img.png", (512, 512))
    mp = SRC / n / "mask.png"
    m = load(mp, (512, 512)) if mp.exists() else None
    for k in range(2):
        ax = axes[0, 2 * j + k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        outline(ax, m)
    axes[0, 2 * j].set_title(TITLES[n], fontsize=6.3, loc="left", pad=2)
    for i, (run, lab) in enumerate(MODELS, start=1):
        seg = load(SRC / n / f"seg_{run}.png", (512, 512))
        cam = load(SRC / n / f"cam_{run}.png", (512, 512))
        ax = axes[i, 2 * j]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        rgba = np.zeros((512, 512, 4)); rgba[..., 0] = 1.0; rgba[..., 1] = 0.85 * (1 - seg); rgba[..., 3] = np.clip(seg * 1.4, 0, 0.9)
        ax.imshow(rgba)
        outline(ax, m, color="#35c2ff")
        ax = axes[i, 2 * j + 1]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_less(cam, 0.05), cmap="jet", vmin=0, vmax=1, alpha=0.5)
        p = IDX[n]["scores"].get(run)
        if p is not None:
            axes[i, 2 * j].text(8, 40, (f"p={p:.2f}" if p >= 0.01 else "p<0.01"), color="white", fontsize=6, bbox=dict(fc="black", alpha=0.55, lw=0, pad=0.8))
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
axes[0, 0].set_ylabel("Image", fontsize=7)
for i, (_, lab) in enumerate(MODELS, start=1):
    axes[i, 0].set_ylabel(lab, fontsize=7)
for j in range(len(CASES)):
    axes[-1, 2 * j].set_xlabel(r"$\sigma(Z)$", fontsize=7)
    axes[-1, 2 * j + 1].set_xlabel("Grad-CAM", fontsize=7)
fig.savefig(OUT / "fig_maps.pdf", bbox_inches="tight", dpi=200)
plt.close(fig)
print("ok", names, [m[0] for m in MODELS])
