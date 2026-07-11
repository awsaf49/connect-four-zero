#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

conda run -n connect4-zero python -m connect4_zero.cli serve --reload &
npm --prefix web run dev &
wait

