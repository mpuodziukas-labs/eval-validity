from evalvalidity import contam

CFG = {"ngram_n": 13, "ngram_flag": 0.05, "shingle_k": 3, "minhash_perms": 128,
       "minhash_seed": 7, "jaccard_dup": 0.7}
W = [f"w{i}" for i in range(40)]


def test_normalize():
    assert contam.normalize("The  Quick, BROWN!") == "the quick brown"
    assert contam.normalize("café x") == contam.normalize("café x")


def test_exact_after_normalization():
    doc = "the quick brown fox jumps over the lazy dog"
    item = "The  Quick, Brown fox jumps over the lazy DOG!"
    assert contam.exact_matches([item, "other text"], [doc]) == [True, False]
    assert contam.exact_matches([item], [doc], norm=lambda s: s) == [False]


def test_ngram_rate():
    text = " ".join(W[:13])
    grams = contam.corpus_grams([text], 13)
    assert contam.ngram_rate(text, grams, 13) == 1.0
    assert contam.ngram_rate(" ".join(reversed(W[:13])), grams, 13) == 0.0
    assert contam.ngram_rate(" ".join(W[:12]), grams, 13) == 0.0


def test_ngram_rate_partial():
    grams = contam.corpus_grams([" ".join(W[:20])], 13)
    item = " ".join(["x%d" % i for i in range(8)] + W[:20])
    assert contam.ngram_rate(item, grams, 13) == 8 / 16


def test_minhash_estimates():
    perms = contam.make_perms(128, 7)
    s = contam.shingles(" ".join(W[:30]), 3)
    near = contam.shingles(" ".join(W[:15] + ["zzz"] + W[16:30]), 3)
    far = contam.shingles(" ".join("q%d" % i for i in range(30)), 3)
    sig = contam.signature(s, perms)
    assert contam.est_jaccard(sig, contam.signature(s, perms)) == 1.0
    assert contam.est_jaccard(sig, contam.signature(far, perms)) < 0.1
    exact = len(s & near) / len(s | near)
    assert abs(contam.est_jaccard(sig, contam.signature(near, perms)) - exact) < 0.15


def test_classify_three_verdicts():
    src = " ".join(W[:30])
    corpus = [{"id": "c0", "text": src}, {"id": "c1", "text": " ".join(W[30:40] * 2)}]
    near = " ".join(W[:15] + ["zzz"] + W[16:30])
    span = " ".join(["x%d" % i for i in range(8)] + W[:20])
    clean = " ".join("y%d" % i for i in range(30))
    items = [{"id": "dup", "text": near}, {"id": "ov", "text": span}, {"id": "ok", "text": clean},
             {"id": "ex", "text": src.upper() + "!!"}]
    out = {r["id"]: r["verdict"] for r in contam.classify_items(items, corpus, CFG)}
    assert out == {"dup": "DUPLICATE", "ov": "OVERLAP", "ok": "CLEAN", "ex": "DUPLICATE"}
