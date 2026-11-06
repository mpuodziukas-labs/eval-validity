"""Plant four defects and one clean control; every defect must be caught by the
oracle suite and the control must pass it."""
import math
import random
from functools import partial

from . import contam, stats


def _mcnemar_all_pairs(a, b):
    n10, n01 = stats.discordant(a, b)
    n = len(a)
    tail = sum(math.comb(n, i) for i in range(min(n10, n01) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _boot_independent(a, b, seed, resamples=10000, level=0.95):
    rng = random.Random(seed)
    n = len(a)
    means = sorted(sum(rng.choices(a, k=n)) / n - sum(rng.choices(b, k=n)) / n
                   for _ in range(resamples))
    return means[int(round((1 - level) / 2 * resamples))], \
        means[int(round((1 + level) / 2 * resamples)) - 1]


def _overlap(n, item, corpus):
    return contam.ngram_rate(item, contam.corpus_grams(corpus, n), n)


REAL = {"mcnemar": stats.mcnemar_exact,
        "boot": stats.paired_bootstrap_ci,
        "overlap": partial(_overlap, contam.DEFAULT_N),
        "exact": contam.exact_matches}


def checks(impl):
    """Return the names of oracle checks the implementation fails."""
    fails = []
    a = [1] * 8 + [0] * 2 + [1] * 45 + [0] * 45
    b = [0] * 8 + [1] * 2 + [1] * 45 + [0] * 45
    if abs(impl["mcnemar"](a, b) - 112 / 1024) > 1e-12:
        fails.append("mcnemar_discordant_only")
    v = [1, 0, 1, 1, 0, 1, 0, 0] * 5
    if impl["boot"](v, list(v), 1, 300) != (0, 0):
        fails.append("bootstrap_pairing")
    ws = ["w%d" % i for i in range(13)]
    fwd, rev = " ".join(ws), " ".join(reversed(ws))
    if impl["overlap"](fwd, [fwd]) != 1.0 or impl["overlap"](rev, [fwd]) != 0.0:
        fails.append("ngram_size")
    doc = "the quick brown fox jumps over the lazy dog"
    if impl["exact"](["The  Quick, Brown fox jumps over the lazy DOG!"], [doc]) != [True]:
        fails.append("normalization")
    return fails


CASES = [
    ("defect_mcnemar_all_pairs", dict(REAL, mcnemar=_mcnemar_all_pairs)),
    ("defect_bootstrap_independent", dict(REAL, boot=_boot_independent)),
    ("defect_ngram_size_1", dict(REAL, overlap=partial(_overlap, 1))),
    ("defect_skip_normalization", dict(REAL, exact=partial(contam.exact_matches,
                                                          norm=lambda s: s))),
    ("control_clean", REAL),
]


def main():
    good = 0
    for name, impl in CASES:
        fails = checks(impl)
        caught = bool(fails)
        ok = caught if name.startswith("defect") else not caught
        good += ok
        print("selftest %-30s %s %s" % (name, "ok" if ok else "WRONG", ",".join(fails) or "-"))
    print("selftest %d/%d" % (good, len(CASES)))
    return 0 if good == len(CASES) else 1
