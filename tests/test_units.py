import numpy as np
from sklearn.metrics import roc_auc_score

from ptx.data.rle import rle_decode, rle_encode
from ptx.eval.metrics import sens_spec_at, threshold_at_specificity
from ptx.eval.size import rim_apex_px
from ptx.eval.stats import cluster_bootstrap, delong_test, holm


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    m = (rng.random((32, 32)) > 0.8).astype(np.uint8)
    for mode in ("relative", "absolute"):
        assert (rle_decode(rle_encode(m, mode), 32, 32, mode) == m).all()
    assert rle_decode("-1", 8, 8).sum() == 0


def test_delong_matches_sklearn_and_null():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s1 = y + rng.normal(0, 1, 400)
    s2 = y + rng.normal(0, 1.5, 400)
    a1, a2, d, p = delong_test(y, s1, s2)
    assert abs(a1 - roc_auc_score(y, s1)) < 1e-9 and abs(a2 - roc_auc_score(y, s2)) < 1e-9
    assert 0 <= p <= 1
    assert delong_test(y, s1, s1)[3] == 1.0


def test_threshold_spec():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(1000), np.ones(100)]
    s = np.r_[rng.random(1000), rng.random(100) + 0.5]
    t = threshold_at_specificity(y, s, 0.95)
    _, sp = sens_spec_at(y, s, t)
    assert sp >= 0.95 and sp < 0.955


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_ci_contains_point():
    x = np.random.default_rng(3).normal(1, 1, 300)
    pt, lo, hi, _ = cluster_bootstrap(lambda i: x[i].mean(), np.arange(300), 500)
    assert lo < pt < hi


def test_size():
    m = np.zeros((100, 100), np.uint8)
    m[0:30, 5:15] = 1
    assert rim_apex_px(m) == (10, 30)
