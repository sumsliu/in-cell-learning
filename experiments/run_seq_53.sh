#!/usr/bin/env bash
# exp_seq: the continual-learning lifecycle, the one experiment that directly
# supports the paper's central claim. Runs on .53 ahead of the round-5 queue.
#   nohup bash experiments/run_seq_53.sh > out/seq.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
PY=.venv/bin/python

while pgrep -f "run_ablation4\.sh" >/dev/null \
   || pgrep -f "exp0_[c]lip" >/dev/null \
   || pgrep -f "exp5_[q]il" >/dev/null; do sleep 30; done

run() {
  name=$1; shift
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY experiments/exp_seq.py "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

BASE="--model Qwen/Qwen3-1.7B-Base --facts-per-task 500 --epochs 24 --rank 64 --replay-frac 0.1"

# minor-version regime: one frozen artifact for the whole sequence
run exp15_seq_minor       $BASE --tasks 4 --mode minor
# with an explicit consolidation (major version) after task 2
run exp15_seq_consolidate $BASE --tasks 4 --mode consolidate --consolidate-after 2

echo "SEQ_53_DONE"
