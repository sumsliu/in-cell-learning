#!/usr/bin/env bash
# Round 7: fair Mistral comparison and the stability observation.
#
# exp23_mistral_a diverged at lr 2e-4 (adapter perplexity 2311 BEFORE any
# merging, clip rate 40%, norm kept 0.40 -- all divergence signatures), while
# CellLoRA on the same model trained cleanly at lr 1e-3, five times higher.
# The bounded reparameterization cannot run away because |fill| < M by
# construction. To claim that fairly we need plain LoRA at a learning rate
# where it is stable, so we sweep down.
#   $1 = gpu index
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${1:-1}"
PY=.venv/bin/python

# wait only for the queue sharing this GPU, not the other one
for u in clora-q6b; do
  while [ "$(systemctl --user is-active $u 2>/dev/null)" = "active" ]; do
    sleep 30
  done
done

run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $(date +%H:%M) $name ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

for LR in 5e-5 2e-5; do
  run "exp25_mistral_a_lr$LR" exp0_clip_rate.py \
      --model mistralai/Mistral-7B-v0.3 --n-facts 1000 --epochs 24 \
      --replay-frac 0.1 --bs 8 --lr "$LR"
done

# same stability question on Qwen3-1.7B, where we can afford the full sweep:
# does plain LoRA break before CellLoRA does?
for LR in 1e-3 3e-3; do
  run "exp26_lora_lr$LR"     exp0_clip_rate.py --n-facts 1000 --epochs 24 \
      --replay-frac 0.1 --bs 16 --lr "$LR"
  run "exp26_cellora_lr$LR"  exp5_qil.py       --n-facts 1000 --epochs 24 \
      --replay-frac 0.1 --rank 16 --tanh-scale 40 --lr "$LR"
done

echo "QUEUE7_DONE"
