"""Seeded SYNTHETIC data: per-item results for three model pairs, an eval set and a
corpus with planted leaks. Nothing here comes from a real model or a real dataset."""
import json
import os
import sys

import random

SYL = ["ka", "lo", "mi", "re", "tu", "sa", "ven", "dor", "pil", "qua", "zen", "fro", "bel",
       "nyx", "gar", "hom", "ixi", "jun", "wex", "yol"]
LABEL = "SYNTHETIC"


def vocab(rng, size=400):
    seen, out = set(), []
    while len(out) < size:
        w = "".join(SYL[rng.randrange(len(SYL))] for _ in range(2 + rng.randrange(2)))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def words(rng, voc, n):
    return [voc[rng.randrange(len(voc))] for _ in range(n)]


def sentence(ws):
    return ws[0].capitalize() + " " + " ".join(ws[1:]) + "."


def pair(seed, n, acc_a, acc_b, name, claim):
    rng = random.Random(seed)
    a, b = [], []
    for _ in range(n):
        base = rng.random()
        a.append(1 if base + 0.15 * (rng.random() - 0.5) < acc_a else 0)
        b.append(1 if base + 0.15 * (rng.random() - 0.5) < acc_b else 0)
    return {"name": name, "claim_significant": claim, "a": a, "b": b}


def build_text(seed=11):
    rng = random.Random(seed)
    voc = vocab(rng)
    corpus = [words(rng, voc, 40) for _ in range(200)]
    near_src = [words(rng, voc, 28) for _ in range(3)]
    items = [(sentence(words(rng, voc, 28)), "none") for _ in range(50)]
    for i in range(3):  # exact after normalization: case, spacing, punctuation changed
        items.append(("  ".join(corpus[i]).upper() + "!!", "exact"))
    for i, src in enumerate(near_src):  # one word replaced
        w = list(src)
        w[14] = "zzq" + str(i)
        items.append((sentence(w), "near"))
    for i in range(4):  # 20 copied words inside fresh text
        items.append((sentence(words(rng, voc, 8) + corpus[3 + i][5:25]), "span"))
    corp = [{"id": "c%03d" % i, "label": LABEL, "text": sentence(w)} for i, w in
            enumerate(corpus + near_src)]
    ev = [{"id": "e%03d" % i, "label": LABEL, "text": t, "planted": p}
          for i, (t, p) in enumerate(items)]
    return ev, corp


def main(outdir="data"):
    os.makedirs(outdir, exist_ok=True)
    ev, corp = build_text()
    pairs = [pair(101, 400, 0.78, 0.70, "real_gap", True),
             pair(202, 400, 0.75, 0.75, "no_gap", False),
             pair(303, 30, 0.80, 0.60, "small_n", True)]
    with open(os.path.join(outdir, "pairs.json"), "w") as f:
        f.write(json.dumps({"label": LABEL, "pairs": pairs}, sort_keys=True) + "\n")
    for name, rows in (("eval.jsonl", ev), ("corpus.jsonl", corp)):
        with open(os.path.join(outdir, name), "w") as f:
            f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:2])
