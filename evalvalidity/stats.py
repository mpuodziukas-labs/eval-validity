"""Paired significance for two models scored on the same items (stdlib only)."""
import math
import random
from statistics import NormalDist


def discordant(a, b):
    if len(a) != len(b):
        raise ValueError("per-item vectors must have equal length")
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    return n10, n01


def mcnemar_exact(a, b):
    """Two-sided exact McNemar: binomial(0.5) on the discordant pairs only."""
    n10, n01 = discordant(a, b)
    n = n10 + n01
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(n10, n01) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired_bootstrap_ci(a, b, seed, resamples=10000, level=0.95):
    """Percentile CI of mean(a) - mean(b); items are resampled jointly."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(resamples))
    lo = means[int(round((1 - level) / 2 * resamples))]
    hi = means[int(round((1 + level) / 2 * resamples)) - 1]
    return lo, hi


def mde(n, p_discordant, alpha=0.05, power=0.8):
    """Smallest accuracy difference detectable with n pairs (normal approximation)."""
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    return z * math.sqrt(p_discordant / n)


def evaluate_pair(a, b, cfg):
    n = len(a)
    n10, n01 = discordant(a, b)
    p = mcnemar_exact(a, b)
    lo, hi = paired_bootstrap_ci(a, b, cfg["seed"], cfg["bootstrap_resamples"], cfg["ci_level"])
    m = mde(n, max(n10 + n01, 1) / n, cfg["alpha"], cfg["power"])
    if m > cfg["max_mde"]:
        verdict = "UNDERPOWERED"
    elif p < cfg["alpha"] and (lo > 0 or hi < 0):
        verdict = "SIGNIFICANT"
    else:
        verdict = "NOT_SIGNIFICANT"
    return {"n": n, "acc_a": round(sum(a) / n, 6), "acc_b": round(sum(b) / n, 6),
            "diff": round((sum(a) - sum(b)) / n, 6), "n10": n10, "n01": n01,
            "p_value": float("%.6g" % p), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
            "mde": round(m, 6), "verdict": verdict}
