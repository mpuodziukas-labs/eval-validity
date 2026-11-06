import json
import os
import pathlib
import subprocess
import sys

from evalvalidity import gate, gen_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CFG = ROOT / "thresholds.json"


def run_gate(data, out):
    return gate.run(str(data), str(CFG), str(out))


def test_shipped_data_fails_gate_for_both_reasons(tmp_path):
    rc = run_gate(DATA, tmp_path / "r.json")
    res = json.loads((tmp_path / "r.json").read_text())
    assert rc == 1 and res["exit"] == 1
    kinds = {f.split(":")[0] for f in res["failures"]}
    assert kinds == {"contamination", "underpowered_claim"}
    verdicts = {p["name"]: p["verdict"] for p in res["pairs"]}
    assert verdicts == {"no_gap": "NOT_SIGNIFICANT", "real_gap": "SIGNIFICANT",
                        "small_n": "UNDERPOWERED"}
    assert res["contamination"]["planted_recovered"] == "10/10"
    assert res["contamination"]["clean_false_positives"] == 0


def test_clean_inputs_pass_gate(tmp_path):
    d = tmp_path / "clean"
    d.mkdir()
    (d / "corpus.jsonl").write_text((DATA / "corpus.jsonl").read_text())
    items = [json.loads(x) for x in (DATA / "eval.jsonl").read_text().splitlines()]
    keep = [i for i in items if i["planted"] == "none"]
    (d / "eval.jsonl").write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in keep))
    pairs = json.loads((DATA / "pairs.json").read_text())
    for p in pairs["pairs"]:
        p["claim_significant"] = p["name"] == "real_gap"
    (d / "pairs.json").write_text(json.dumps(pairs, sort_keys=True) + "\n")
    assert run_gate(d, tmp_path / "r.json") == 0


def test_results_are_byte_identical(tmp_path):
    run_gate(DATA, tmp_path / "a.json")
    run_gate(DATA, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_data_regenerates_identically(tmp_path):
    gen_data.main(str(tmp_path))
    for name in ("pairs.json", "eval.jsonl", "corpus.jsonl"):
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()


def test_all_data_labeled_synthetic():
    assert json.loads((DATA / "pairs.json").read_text())["label"] == "SYNTHETIC"
    for name in ("eval.jsonl", "corpus.jsonl"):
        lines = (DATA / name).read_text().splitlines()
        assert lines and all(json.loads(x)["label"] == "SYNTHETIC" for x in lines)


def test_readme_table_matches_measured(tmp_path):
    run_gate(DATA, tmp_path / "r.json")
    res = json.loads((tmp_path / "r.json").read_text())
    readme = (ROOT / "README.md").read_text()
    lines = gate.table_lines(res)
    assert len(lines) >= 8
    for line in lines:
        assert line in readme


def test_selftest_cli():
    p = subprocess.run([sys.executable, "-P", "-m", "evalvalidity", "--selftest"], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert p.stdout.strip().splitlines()[-1] == "selftest 5/5"
