"""Contamination checks between eval items and a reference corpus (stdlib only)."""
import hashlib
import random
import re
import unicodedata

DEFAULT_N = 13
_P = (1 << 61) - 1


def normalize(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def exact_matches(items, corpus, norm=normalize):
    seen = {hashlib.sha256(norm(c).encode()).hexdigest() for c in corpus}
    return [hashlib.sha256(norm(i).encode()).hexdigest() in seen for i in items]


def corpus_grams(texts, n):
    grams = set()
    for t in texts:
        w = normalize(t).split()
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return grams


def ngram_rate(text, grams, n):
    """Fraction of the item's word n-grams that occur in the corpus."""
    w = normalize(text).split()
    mine = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return sum(g in grams for g in mine) / len(mine) if mine else 0.0


def shingles(text, k):
    w = normalize(text).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def make_perms(num, seed):
    rng = random.Random(seed)
    return [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num)]


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def signature(shs, perms):
    hs = [_h(s) for s in shs]
    return [min((a * h + b) % _P for h in hs) for a, b in perms]


def est_jaccard(s1, s2):
    return sum(x == y for x, y in zip(s1, s2)) / len(s1)


def classify_items(items, corpus, cfg):
    """items/corpus: lists of {"id","text"}. Returns one record per item."""
    n, k = cfg["ngram_n"], cfg["shingle_k"]
    texts = [c["text"] for c in corpus]
    grams = corpus_grams(texts, n)
    perms = make_perms(cfg["minhash_perms"], cfg["minhash_seed"])
    csigs = [signature(s, perms) for s in (shingles(t, k) for t in texts) if s]
    exact = exact_matches([i["text"] for i in items], texts)
    out = []
    for item, ex in zip(items, exact):
        rate = ngram_rate(item["text"], grams, n)
        sh = shingles(item["text"], k)
        jac = max((est_jaccard(signature(sh, perms), c) for c in csigs), default=0.0) if sh else 0.0
        if ex or jac >= cfg["jaccard_dup"]:
            verdict = "DUPLICATE"
        elif rate >= cfg["ngram_flag"]:
            verdict = "OVERLAP"
        else:
            verdict = "CLEAN"
        out.append({"id": item["id"], "verdict": verdict, "exact": ex,
                    "ngram_rate": round(rate, 6), "max_jaccard": round(jac, 6)})
    return out
