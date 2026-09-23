"""Pneumothorax size from a binary mask (component C5, used for A4 and A6).

Two guideline-style measurements, computed identically on the reference mask and the
predicted mask so that agreement (weighted kappa) measures the model, not the rule:

  rim_px   largest horizontal air-rim width (max over rows of the per-side run of mask pixels)
           -> BTS-style: 'large' if rim >= 20 mm
  apex_px  largest vertical air-cap thickness in the upper 40% of the image
           -> ACCP-style: 'large' if apex-to-cupola >= 30 mm

When pixel spacing is unknown (PNG-only datasets) the size class falls back to
tertiles of mask area fraction, with cut-points taken from the reference masks of
that dataset. The Light index needs lung and hemithorax segmentation and is not
implemented; the paper must say so.
"""
from __future__ import annotations

import numpy as np


def rim_apex_px(mask: np.ndarray):
    m = mask.astype(bool)
    if not m.any():
        return 0, 0
    h, w = m.shape
    rim = max(int(m[:, : w // 2].sum(1).max()), int(m[:, w // 2:].sum(1).max()))
    apex = int(m[: int(0.4 * h)].sum(0).max()) if m[: int(0.4 * h)].any() else 0
    return rim, apex


def size_features(mask: np.ndarray, spacing_mm: float | None):
    rim, apex = rim_apex_px(mask)
    area_frac = float(mask.astype(bool).mean())
    sp = spacing_mm if spacing_mm is not None and spacing_mm == spacing_mm else np.nan
    return {"rim_px": rim, "apex_px": apex, "area_frac": area_frac,
            "rim_mm": rim * sp, "apex_mm": apex * sp}


def bts_class(rim_mm):
    return np.where(np.isnan(rim_mm), -1, (np.asarray(rim_mm) >= 20).astype(int))


def accp_class(apex_mm):
    return np.where(np.isnan(apex_mm), -1, (np.asarray(apex_mm) >= 30).astype(int))


def tertile_class(area_frac, cuts):
    return np.digitize(np.asarray(area_frac), cuts)  # 0 small, 1 medium, 2 large


def is_small(df, ref_cuts=None):
    """'Small' pneumothorax per the pre-registered rule: BTS rim < 2 cm when spacing is known,
    otherwise lowest tertile of reference area fraction. Uses REFERENCE mask features (gt_*)."""
    small = np.zeros(len(df), dtype=bool)
    has_mm = df["gt_rim_mm"].notna().values
    small[has_mm] = df["gt_rim_mm"].values[has_mm] < 20
    if (~has_mm).any():
        cuts = ref_cuts if ref_cuts is not None else np.quantile(df.loc[~has_mm, "gt_area_frac"], [1 / 3, 2 / 3])
        small[~has_mm] = df["gt_area_frac"].values[~has_mm] <= cuts[0]
    return small
