#!/usr/bin/env bash
# exp14: WHERE does knowledge live? Module-type and depth partitions under
# the QIL r64 champion recipe (structural invariance -> clean attribution).
# GPU1 on .145. Launch:
#   systemd-run --user --unit clora-where --collect bash -c \
#     "cd ~/clora && bash experiments/run_where.sh > out/where.log 2>&1"
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python

run5() {
  name=$1; shift
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $name: $* ==="
  $PY experiments/exp5_qil.py "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

BASE="--model Qwen/Qwen3-1.7B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3"

# module-type axis
run5 exp14_mlp    $BASE --target-filter "(gate_proj|up_proj|down_proj)"
run5 exp14_attn   $BASE --target-filter "(q_proj|k_proj|v_proj|o_proj)"
run5 exp14_down   $BASE --target-filter "down_proj"
run5 exp14_gateup $BASE --target-filter "(gate_proj|up_proj)"
# depth axis (MLP only; Qwen3-1.7B has 28 layers -> thirds)
run5 exp14_early  $BASE --target-filter "layers\.[0-8]\..*(gate_proj|up_proj|down_proj)"
run5 exp14_mid    $BASE --target-filter "layers\.(9|1[0-8])\..*(gate_proj|up_proj|down_proj)"
run5 exp14_late   $BASE --target-filter "layers\.(19|2[0-7])\..*(gate_proj|up_proj|down_proj)"

echo "WHERE_DONE"
