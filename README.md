# eval-validity

[![CI](https://github.com/mpuodziukas-labs/eval-validity/actions/workflows/ci.yml/badge.svg)](https://github.com/mpuodziukas-labs/eval-validity/actions/workflows/ci.yml)

Quickstart: `bash scripts/verify.sh` prints `GREEN` (details: [Reproduce](#reproduce)).

Decide whether an eval result can be trusted before anyone cites it. Standard library only, no network, no model call in CI.

All data in this repository is SYNTHETIC. It is generated from fixed seeds by `evalvalidity/gen_data.py` and every record carries `"label": "SYNTHETIC"`. No real model or real dataset was measured.

## Honesty Statement

All data here is synthetic, generated from fixed seeds, and every record is labeled SYNTHETIC. The leaks are planted by the author of the detector, and no real model or dataset was measured. The numbers show that the statistics and the contamination checks recover what was planted and that the gate exits as designed. They do not show what real leaks or real model comparisons look like. See Limitations.

## What it checks

1. Model A vs model B on per-item 0/1 correctness: exact McNemar test (binomial on the discordant pairs), a seeded paired bootstrap CI on the accuracy difference (10000 resamples), and the minimum detectable effect (MDE) for the observed n and discordant rate. Verdict: `SIGNIFICANT`, `NOT_SIGNIFICANT` or `UNDERPOWERED`. A pair is `UNDERPOWERED` when its MDE exceeds `max_mde` in `thresholds.json`, whatever its p-value.
2. Contamination between an eval set and a reference corpus: exact hash after normalization, per-item word 13-gram overlap rate, and near-duplicate detection with a hand-written MinHash (128 permutations, word 3-shingles, Jaccard threshold 0.7). Verdict per item: `CLEAN`, `OVERLAP` or `DUPLICATE`.
3. Gate (`thresholds.json`): exit 1 if contaminated items exceed `contaminated_floor`, or if a pair that claims significance is `UNDERPOWERED`.

## Measured results

Produced by the command below. The tests fail if these tables drift from the measured output.

| pair | n | acc A | acc B | diff | 95% CI | McNemar p | MDE | verdict |
|---|---|---|---|---|---|---|---|---|
| real_gap | 400 | 0.7575 | 0.6700 | +0.0875 | [+0.0575, +0.1175] | 1.048e-08 | 0.0448 | SIGNIFICANT |
| no_gap | 400 | 0.7950 | 0.7925 | +0.0025 | [-0.0175, +0.0225] | 1 | 0.0289 | NOT_SIGNIFICANT |
| small_n | 30 | 0.6000 | 0.4333 | +0.1667 | [+0.0333, +0.3000] | 0.0625 | 0.2088 | UNDERPOWERED |

| eval items | CLEAN | OVERLAP | DUPLICATE | planted recovered | clean false positives |
|---|---|---|---|---|---|
| 60 | 50 | 4 | 6 | 10/10 | 0 |

The shipped gate run exits 1 on purpose: the data plants 10 leaks (3 exact after normalization, 3 one-word near-duplicates, 4 items with a 20-word copied span) and `small_n` carries a significance claim. `scripts/verify.sh` asserts that exit code, then checks two runs write byte-identical `results.json`. `python3 -m evalvalidity --selftest` plants four defects (McNemar over all pairs, independent bootstrap resampling, n-gram size 1, no normalization before hashing) plus one clean control and prints `selftest 5/5`.

## Limitations

- The data is synthetic and the leaks are planted by the same author as the detector. "Planted recovered 10/10" shows the detector finds what it was built to find, not what real leaks look like.
- The accuracy columns come from a generator with a shared per-item difficulty term plus noise. Real model errors may be correlated differently, which changes the discordant rate and so the MDE.
- MDE uses a normal approximation with the observed discordant rate, so it is itself an estimate and is noisy at small n.
- The `UNDERPOWERED` rule is conservative: a large observed gap at small n is never called significant. `max_mde` (0.08) is a project choice, not a standard.
- One seed per pair. The `no_gap` pair is non-significant for this seed; a different seed would be significant about 5 percent of the time by design of the test.
- Contamination checks are lexical. Paraphrased leaks, translated leaks and leaks through a corpus you do not hold are invisible. Items shorter than 13 words have no 13-grams and get an overlap rate of 0.
- MinHash is an estimate (about 0.035 standard error at Jaccard 0.8 with 128 permutations); items near the 0.7 threshold can flip with the seed.
- Per-item correctness is assumed independent across items. Clustered items (several questions on one passage) make the p-values too small.

## Reproduce

```
bash scripts/verify.sh
```

Needs Python 3.8+ and pytest; scipy is optional (one test cross-checks the McNemar p-values against `scipy.stats.binomtest` and skips without it); shellcheck is required by the script.
