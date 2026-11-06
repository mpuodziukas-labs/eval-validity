"""Gate: exit 1 on contamination above the floor or a significance claim on an
UNDERPOWERED pair. Output is deterministic (no timestamps, no paths)."""
import argparse
import json
import os
import sys

from . import contam, stats

EXPECTED = {"exact": "DUPLICATE", "near": "DUPLICATE", "span": "OVERLAP"}


def _jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def run(data_dir, config_path, out_path):
    with open(config_path) as f:
        cfg = json.load(f)
    with open(os.path.join(data_dir, "pairs.json")) as f:
        pairs = json.load(f)["pairs"]
    items = _jsonl(os.path.join(data_dir, "eval.jsonl"))
    corpus = _jsonl(os.path.join(data_dir, "corpus.jsonl"))
    failures, prs = [], []
    for p in pairs:
        r = stats.evaluate_pair(p["a"], p["b"], cfg)
        r["name"], r["claim_significant"] = p["name"], p["claim_significant"]
        prs.append(r)
        if r["claim_significant"] and r["verdict"] == "UNDERPOWERED":
            failures.append("underpowered_claim: %s (n=%d, mde=%.4f)" % (p["name"], r["n"], r["mde"]))
    recs = contam.classify_items(items, corpus, cfg)
    counts = {v: sum(r["verdict"] == v for r in recs) for v in ("CLEAN", "OVERLAP", "DUPLICATE")}
    planted = [(i, r) for i, r in zip(items, recs) if i["planted"] != "none"]
    ok = sum(r["verdict"] == EXPECTED[i["planted"]] for i, r in planted)
    bad = counts["OVERLAP"] + counts["DUPLICATE"]
    if bad > cfg["contaminated_floor"]:
        failures.append("contamination: %d items above floor %d" % (bad, cfg["contaminated_floor"]))
    res = {"label": "SYNTHETIC", "pairs": prs, "failures": failures, "exit": 1 if failures else 0,
           "contamination": {"counts": counts, "contaminated": bad,
                             "floor": cfg["contaminated_floor"], "n_items": len(items),
                             "planted_recovered": "%d/%d" % (ok, len(planted)),
                             "clean_false_positives": sum(
                                 r["verdict"] != "CLEAN" for i, r in zip(items, recs)
                                 if i["planted"] == "none"),
                             "flagged": [r for r in recs if r["verdict"] != "CLEAN"]}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(res, sort_keys=True, indent=2) + "\n")
    return res["exit"]


def table_lines(res):
    rows = ["| pair | n | acc A | acc B | diff | 95% CI | McNemar p | MDE | verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in res["pairs"]:
        rows.append("| %s | %d | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %.4g | %.4f | %s |" % (
            r["name"], r["n"], r["acc_a"], r["acc_b"], r["diff"], r["ci_lo"], r["ci_hi"],
            r["p_value"], r["mde"], r["verdict"]))
    c = res["contamination"]
    rows += ["", "| eval items | CLEAN | OVERLAP | DUPLICATE | planted recovered | clean false positives |",
             "|---|---|---|---|---|---|",
             "| %d | %d | %d | %d | %s | %d |" % (
                 c["n_items"], c["counts"]["CLEAN"], c["counts"]["OVERLAP"],
                 c["counts"]["DUPLICATE"], c["planted_recovered"], c["clean_false_positives"])]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evalvalidity")
    ap.add_argument("--config", default="thresholds.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        from . import selftest
        return selftest.main()
    rc = run(args.data, args.config, args.out)
    with open(args.out) as f:
        res = json.load(f)
    if args.table:
        print("\n".join(table_lines(res)))
    for msg in res["failures"]:
        print("FAIL " + msg)
    print("gate exit %d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
