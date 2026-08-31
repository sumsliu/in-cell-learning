#!/usr/bin/env bash
# Multi-seed reruns of the invariant frontier (QIL / A+ / B) for CIs.
# Pinned to GPU1 on .145; coexists with exp12's GPU0 training.
#   systemd-run --user --unit clora-seeds --collect bash -c \
#     "cd ~/clora && bash experiments/run_seeds.sh > out/seeds.log 2>&1"
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python

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

for SEED in 1 2; do
  run5 "exp5b_qil_s$SEED" --n-facts 1000 --epochs 24 --replay-frac 0.3 \
       --tanh-scale 40 --lr 1e-3 --seed "$SEED"
  run0 "exp1c_s$SEED" --n-facts 1000 --epochs 24 --replay-frac 0.1 \
       --heal-epochs 4 --heal-lr 2e-5 --seed "$SEED"
  run0 "exp3_s$SEED" --n-facts 1000 --epochs 24 --replay-frac 0.3 \
       --dense-only --heal-lr 2e-5 --seed "$SEED"
done

echo SEEDS_DONE
