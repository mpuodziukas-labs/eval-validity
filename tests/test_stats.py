import random

import pytest

from evalvalidity import stats

CFG = {"alpha": 0.05, "power": 0.8, "max_mde": 0.08, "bootstrap_resamples": 2000,
       "ci_level": 0.95, "seed": 1}


def vecs(n10, n01, both, neither):
    a = [1] * n10 + [0] * n01 + [1] * both + [0] * neither
    b = [0] * n10 + [1] * n01 + [1] * both + [0] * neither
    return a, b


def test_mcnemar_known_value():
    a, b = vecs(8, 2, 45, 45)
    assert stats.mcnemar_exact(a, b) == pytest.approx(112 / 1024, abs=1e-15)


def test_mcnemar_ignores_concordant_pairs():
    a1, b1 = vecs(8, 2, 0, 0)
    a2, b2 = vecs(8, 2, 500, 500)
    assert stats.mcnemar_exact(a1, b1) == stats.mcnemar_exact(a2, b2)


def test_mcnemar_edges():
    assert stats.mcnemar_exact([1, 0], [1, 0]) == 1.0
    a, b = vecs(10, 0, 0, 0)
    assert stats.mcnemar_exact(a, b) == pytest.approx(2 / 1024, abs=1e-15)
    a, b = vecs(5, 5, 0, 0)
    assert stats.mcnemar_exact(a, b) == 1.0


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        stats.mcnemar_exact([1, 0], [1])


def test_mcnemar_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    rng = random.Random(5)
    for _ in range(25):
        n10, n01 = rng.randrange(0, 60), rng.randrange(0, 60)
        if n10 + n01 == 0:
            continue
        a, b = vecs(n10, n01, 3, 3)
        ref = st.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue
        assert stats.mcnemar_exact(a, b) == pytest.approx(ref, abs=1e-9)


def test_bootstrap_preserves_pairing():
    a = [1, 0, 1, 1, 0, 1, 0, 0] * 5
    assert stats.paired_bootstrap_ci(a, list(a), seed=3, resamples=300) == (0.0, 0.0)


def test_bootstrap_deterministic_and_brackets_diff():
    a, b = vecs(30, 10, 100, 60)
    ci1 = stats.paired_bootstrap_ci(a, b, seed=9, resamples=2000)
    ci2 = stats.paired_bootstrap_ci(a, b, seed=9, resamples=2000)
    assert ci1 == ci2
    diff = (sum(a) - sum(b)) / len(a)
    assert ci1[0] < diff < ci1[1]


def test_mde_value_and_monotone():
    assert stats.mde(400, 0.2) == pytest.approx(0.062647, abs=1e-5)
    assert stats.mde(100, 0.2) > stats.mde(400, 0.2) > stats.mde(1600, 0.2)


def test_verdict_significant():
    a, b = vecs(80, 20, 200, 100)
    r = stats.evaluate_pair(a, b, CFG)
    assert r["verdict"] == "SIGNIFICANT"
    assert r["n10"] == 80 and r["n01"] == 20


def test_verdict_not_significant():
    a, b = vecs(30, 28, 200, 142)
    assert stats.evaluate_pair(a, b, CFG)["verdict"] == "NOT_SIGNIFICANT"


def test_verdict_underpowered_even_when_p_small():
    a, b = vecs(12, 1, 10, 7)
    r = stats.evaluate_pair(a, b, CFG)
    assert r["p_value"] < 0.05
    assert r["mde"] > CFG["max_mde"]
    assert r["verdict"] == "UNDERPOWERED"
