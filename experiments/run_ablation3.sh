#!/usr/bin/env bash
# Round 3: tier-1 completion + first tier-2 axes + capacity probe.
# Waits for round 2 (run_ablation.sh) and any live experiment, then runs
# serially. Idempotent. Usage:
#   nohup bash experiments/run_ablation3.sh > out/ablation3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com
PY=.venv/bin/python

while pgrep -f "run_ablation\.sh" >/dev/null \
   || pgrep -f "exp0_[c]lip" >/dev/null \
   || pgrep -f "exp5_[q]il" >/dev/null; do sleep 30; done

run0() {
  name=$1; shift
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY experiments/exp0_clip_rate.py "$@" --out "out/$name.json" \
      > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}
run5() {
  name=$1; shift
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY experiments/exp5_qil.py "$@" --out "out/$name.json" \
      > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# QIL-LoRA, corrected effective step size (was 100x too small at s=4/lr=2e-4)
run5 exp5b_qil      --n-facts 1000 --epochs 24 --replay-frac 0.3 --tanh-scale 40 --lr 1e-3
# replay-frac axis: 10%
run0 exp1c_replay10 --n-facts 1000 --epochs 24 --replay-frac 0.1 --heal-epochs 4 --heal-lr 2e-5
# radius axis: rho=0.5 (half the in-bin room)
run0 exp7_rho05     --n-facts 1000 --epochs 24 --replay-frac 0.3 --heal-epochs 4 --heal-lr 2e-5 --radius 0.5
# capacity probe: 10k facts (~4h; replay auto-switches to wikitext-103)
run0 exp6_10k       --n-facts 10000 --epochs 24 --replay-frac 0.3 --heal-epochs 4 --heal-lr 2e-5

echo "ABLATION3_DONE"
