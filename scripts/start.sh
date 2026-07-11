#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

if [[ ! -f web/dist/index.html ]]; then
  npm --prefix web run build
fi

exec conda run -n connect4-zero python -m connect4_zero.cli serve

