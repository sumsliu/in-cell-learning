#!/usr/bin/env bash
# .53 is free: the 10^5 capacity point, now that the fact generator no longer
# hangs (it was asked for 100k unique names from a 90k-name space and spun
# for four hours). Wide names raise the space by an order of magnitude.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
PY=.venv/bin/python
while pgrep -f "exp0_[c]lip" >/dev/null || pgrep -f "exp5_[q]il" >/dev/null \
   || pgrep -f "exp_[cs]" >/dev/null; do sleep 30; done
run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $(date +%H:%M) $name ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}
# capacity at 10^5, matched to the 10^3/10^4 A+ series (rehearsal 0.3)
run exp22_100k exp0_clip_rate.py --n-facts 100000 --epochs 12 --rank 64 \
    --alpha 128 --replay-frac 0.3 --heal-epochs 2 --heal-lr 2e-5 \
    --probe-cap 9000
# a second sequential seed: the decay law currently rests on one sequence
run exp27_seq_s1 exp_seq.py --model Qwen/Qwen3-1.7B-Base --tasks 4 \
    --facts-per-task 500 --epochs 24 --rank 64 --replay-frac 0.1 --seed 1
# multi-party composition: two disjoint fact sets, two independent residuals
# against one shared artifact, summed and clamped once
run exp28_compose exp_compose.py --n-facts 500 --epochs 24 --replay-frac 0.1

echo "Q8_DONE"
