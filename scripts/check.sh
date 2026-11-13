#!/usr/bin/env bash
# Every recipe check, in the order a pull request runs them: lint, publication scan, links, tests.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0
ruff check . || fail=1
ruff format --check scripts tests || fail=1
bash scripts/check-public.sh || fail=1
python3 scripts/check_links.py || fail=1
python3 -m pytest -q || fail=1
[[ $fail -eq 0 ]] && echo "check: all clean" || echo "check: FAILED"
exit $fail
