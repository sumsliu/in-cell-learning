#!/usr/bin/env bash
# .145 GPU1 round 5: 4B A+/B (8-bit Adam), 8B A/QIL, geometry at 4B.
# Waits for the clora-seeds unit. Launch via systemd-run:
#   systemd-run --user --unit clora-q5b --collect bash -c \
#     "cd ~/clora && bash experiments/run_queue5_145b.sh > out/q5b.log 2>&1"
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python

for unit in clora-where clora-seeds; do
  while [ "$(systemctl --user is-active $unit 2>/dev/null)" = "active" ]; do
    sleep 30
  done
done

run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# scale ladder: 4B A+ and B (dense needs 8-bit Adam on 80G)
run exp21_aplus_4b exp0_clip_rate.py --model Qwen/Qwen3-4B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --heal-epochs 4 --heal-lr 2e-5 --heal-optim adam8bit --bs 8
run exp21_b_4b     exp0_clip_rate.py --model Qwen/Qwen3-4B-Base --n-facts 1000 --epochs 24 --replay-frac 0.3 --dense-only --heal-lr 2e-5 --heal-optim adam8bit --bs 8

# scale ladder: 8B A + QIL (dense heal does not fit; A/QIL do)
run exp24_a_8b   exp0_clip_rate.py --model Qwen/Qwen3-8B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --bs 8
run exp24_qil_8b exp5_qil.py       --model Qwen/Qwen3-8B-Base --n-facts 1000 --epochs 24 --replay-frac 0.3 --tanh-scale 40 --lr 1e-3 --bs 8

# geometry at 4B (radius vs scale)
run exp_geom_4b exp_geom.py --model Qwen/Qwen3-4B-Base --pert-seeds 1

echo "QUEUE5_145B_DONE"
