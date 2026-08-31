#!/usr/bin/env bash
# 27B CellLoRA, third attempt. The first two failed for different reasons:
# fp32 fill materialization (fixed with bf16), then autograd retention
# (fixed with gradient checkpointing), then simple GPU contention -- the
# previous launch shared a card with the 8B ladder and lost 54 GB to it.
# This one waits for BOTH queues so it gets a card to itself.
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python

for u in clora-q6a clora-q7b; do
  while [ "$(systemctl --user is-active $u 2>/dev/null)" = "active" ]; do
    sleep 60
  done
done
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | sort -n | tail -1 | awk '{exit ($1 > 4000) ? 0 : 1}'; do sleep 60; done

[ -f out/exp13_qil_27b.json ] || {
  echo "=== $(date +%H:%M) exp13_qil_27b ==="
  $PY experiments/exp5_qil.py --model Qwen/Qwen3.8-27B --n-facts 1000 \
      --epochs 8 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 \
      --bs 4 --margin 0.02 --map-dtype float16 --skip-anchor-eval \
      --eval-dtype bfloat16 --checkpoint-fill \
      --out out/exp13_qil_27b.json > out/exp13_qil_27b.log 2>&1
  echo "=== exit $? ==="
}
echo "Q10_DONE"
