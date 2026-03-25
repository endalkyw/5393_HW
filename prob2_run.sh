#!/usr/bin/env bash
set -euo pipefail

# paths (edit as needed)
R="lambda.r"
IN="lambda.in"
OUTDIR="out_lambda"

# knobs
N=1000
JOBS=$(nproc)          # Linux; on mac use: sysctl -n hw.ncpu
MAX_TIME=50000
MAX_EVENTS=200000

python3 problem_3.py \
  --r "$R" --in "$IN" \
  --moi-range "1:10" \
  --N "$N" \
  --jobs "$JOBS" \
  --outdir "$OUTDIR" \
  --max-time "$MAX_TIME" \
  --max-events "$MAX_EVENTS"