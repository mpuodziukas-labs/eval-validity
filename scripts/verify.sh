#!/usr/bin/env bash
# Runs the tests, the gate twice (results must be byte-identical), the selftest and shellcheck.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

python3 -m pytest -q
mkdir -p out

rc1=0
python3 -m evalvalidity --out out/r1.json >/dev/null || rc1=$?
rc2=0
python3 -m evalvalidity --out out/r2.json >/dev/null || rc2=$?
echo "gate rc1=$rc1 rc2=$rc2 (1 expected: the shipped data holds planted leaks)"
if [ "$rc1" -ne 1 ] || [ "$rc2" -ne 1 ]; then
  echo "RED: gate did not reject the planted leaks"
  exit 1
fi
cmp out/r1.json out/r2.json
echo "results.json byte-identical"

python3 -m evalvalidity --selftest
shellcheck scripts/verify.sh
echo "shellcheck rc 0"
echo GREEN
