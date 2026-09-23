"""Statistical tests used by the acceptance checker (pre-registered in PREREGISTRATION.md)."""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- DeLong (paired)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    """Sun & Xu (2014) fast DeLong. preds_sorted: (k, n) with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.atleast_2d(np.cov(v01)), np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, s1, s2):
    """Two-sided paired DeLong test. Returns (auc1, auc2, diff, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    P = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    aucs, cov = _fast_delong(P, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(diff), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap(stat_fn, clusters, n_boot=2000, seed=0):
    """Resample clusters (patients) with replacement; stat_fn(idx) -> float. Returns (point, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    point = stat_fn(np.arange(len(clusters)))
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        out[b] = stat_fn(idx)
    out = out[~np.isnan(out)]
    lo, hi = np.percentile(out, [2.5, 97.5]) if len(out) else (np.nan, np.nan)
    return float(point), float(lo), float(hi), out


def boot_p_two_sided(samples, null=0.0):
    """Percentile-bootstrap two-sided p-value for H0: statistic == null."""
    if len(samples) == 0:
        return np.nan
    p = 2 * min((samples <= null).mean(), (samples >= null).mean())
    return float(min(1.0, max(p, 1.0 / len(samples))))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def noninferiority(samples_diff, margin):
    """One-sided: lower 95% bound of (new - reference) > margin (margin negative, e.g. -0.01)."""
    lo = np.percentile(samples_diff, 5)
    return bool(lo > margin), float(lo)
