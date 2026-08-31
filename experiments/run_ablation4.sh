#!/usr/bin/env bash
# Round 4: attack the two diagnosed bottlenecks (exposure, rank) with the
# best-known recipe (fresh replay 10%, heal). All runs carry cross-domain
# LAMBADA PPL. Waits for anything already on the GPU. Idempotent.
#   nohup bash experiments/run_ablation4.sh > out/ablation4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com
PY=.venv/bin/python

while pgrep -f "run_ablation3\.sh" >/dev/null \
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

# exposure axis: 4x epochs at 1k, best recipe
run0 exp8_e96      --n-facts 1000  --epochs 96 --replay-frac 0.1 --heal-epochs 8  --heal-lr 2e-5
# QIL rank axis (cheap): does r64 push the low-drift frontier up?
run5 exp10_qil_r64 --n-facts 1000  --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3
# recipe-v2 at 10k: rank 64 + replay 10% + heal 12 (compound push, not a clean ablation)
run0 exp9_r64_10k  --n-facts 10000 --epochs 24 --replay-frac 0.1 --rank 64 --alpha 128 --heal-epochs 12 --heal-lr 2e-5

echo "ABLATION4_DONE"
