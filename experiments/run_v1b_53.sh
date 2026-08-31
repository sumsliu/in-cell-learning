#!/usr/bin/env bash
# Table-1 consistency: the dense arms were run at 30% rehearsal while the
# low-rank arms are at 10%, and exp16 showed rehearsal is worth ~11 recall
# points and ~13 LAMBADA points for QIL -- far too large to leave mixed
# inside one table. These runs give the dense arms a matched 10% row.
# The 30% runs stay: they are the matched pair used for the capacity series
# and for the paired invariance-cost test.
#   nohup bash experiments/run_v1b_53.sh > out/v1b.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
PY=.venv/bin/python

while pgrep -f "run_v1_53\.sh" >/dev/null \
   || pgrep -f "exp0_[c]lip" >/dev/null \
   || pgrep -f "exp5_[q]il" >/dev/null \
   || pgrep -f "exp_[s]eq" >/dev/null; do sleep 30; done

run() {
  name=$1; shift
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $(date +%H:%M) $name ==="
  $PY experiments/exp0_clip_rate.py "$@" --out "out/$name.json" \
      > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

for S in 0 1 2; do
  run "exp18_b_replay10_s$S" --n-facts 1000 --epochs 24 --replay-frac 0.1 \
      --dense-only --heal-lr 2e-5 --seed "$S"
done

echo "V1B_DONE"
