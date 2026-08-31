#!/usr/bin/env bash
# .53 (4090-48G) round 5: geometry figure, 0.6B frontier, 4B QIL, 100k capacity.
# Waits for round 4. Idempotent.
#   nohup bash experiments/run_queue5_53.sh > out/queue5.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
PY=.venv/bin/python

while pgrep -f "run_ablation4\.sh" >/dev/null \
   || pgrep -f "run_v1_53\.sh" >/dev/null \
   || pgrep -f "exp0_[c]lip" >/dev/null \
   || pgrep -f "exp5_[q]il" >/dev/null \
   || pgrep -f "exp_[g]eom" >/dev/null \
   || pgrep -f "exp_[s]eq" >/dev/null; do sleep 30; done

run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# geometry / safe-radius figure (M1) at 1.7B
run exp_geom_1p7b exp_geom.py --model Qwen/Qwen3-1.7B-Base

# scale ladder: 0.6B frontier (all three invariant methods)
run exp20_qil_0p6   exp5_qil.py       --model Qwen/Qwen3-0.6B-Base --n-facts 1000 --epochs 24 --replay-frac 0.3 --tanh-scale 40 --lr 1e-3
run exp20_aplus_0p6 exp0_clip_rate.py --model Qwen/Qwen3-0.6B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --heal-epochs 4 --heal-lr 2e-5
run exp20_b_0p6     exp0_clip_rate.py --model Qwen/Qwen3-0.6B-Base --n-facts 1000 --epochs 24 --replay-frac 0.3 --dense-only --heal-lr 2e-5

# 4B QIL (fits 48G; 4B A+/B need the A100s)
run exp21_qil_4b    exp5_qil.py       --model Qwen/Qwen3-4B-Base --n-facts 1000 --epochs 24 --replay-frac 0.3 --tanh-scale 40 --lr 1e-3 --bs 8

# capacity: 100k facts, recipe-v2, overnight (~13h); artifacts saved for codec
run exp22_100k      exp0_clip_rate.py --n-facts 100000 --epochs 12 --rank 64 --alpha 128 --replay-frac 0.1 --heal-epochs 2 --heal-lr 2e-5 --probe-cap 9000 --save-merged

echo "QUEUE5_53_DONE"
