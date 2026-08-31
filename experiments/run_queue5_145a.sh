#!/usr/bin/env bash
# .145 GPU0 round 5: 27B QIL, Mistral family generality, then (expensive)
# 27B e24. Ordered cheapest-evidence-first: each item that finishes adds a
# paper claim, so an interruption still leaves the most valuable points done.
# Waits for whatever 27B run is currently active. Launch via systemd-run:
#   systemd-run --user --unit clora-q5a --collect bash -c \
#     "cd ~/clora && bash experiments/run_queue5_145a.sh > out/q5a.log 2>&1"
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python

for unit in clora-exp12s clora-exp12r clora-exp12; do
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

# 1. QIL-LoRA at 27B: structural invariance on a hybrid linear-attention model.
#    bf16 eval is mandatory here (fp32 rebuild of 27B would need ~100 GiB).
run exp13_qil_27b exp5_qil.py --model Qwen/Qwen3.8-27B --n-facts 1000 \
    --epochs 8 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 --bs 4 \
    --margin 0.02 --map-dtype float16 --skip-anchor-eval --eval-dtype bfloat16

# 2. family generality: Mistral-7B (different lineage, standard attention)
run exp23_mistral_a   exp0_clip_rate.py --model mistralai/Mistral-7B-v0.3 \
    --n-facts 1000 --epochs 24 --replay-frac 0.1 --bs 8 --heal-epochs 4 \
    --heal-lr 2e-5 --heal-optim adam8bit
run exp23_mistral_qil exp5_qil.py --model mistralai/Mistral-7B-v0.3 \
    --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 \
    --lr 1e-3 --bs 8

# 3. expensive last: 27B at 24 epochs, single-GPU eval (sharding a hybrid
#    linear-attention model across GPUs produced garbage logits in exp12)
run exp12b_27b_e24 exp0_clip_rate.py --model Qwen/Qwen3.8-27B --n-facts 1000 \
    --epochs 24 --replay-frac 0.1 --bs 8 --margin 0.02 --eval-dtype bfloat16 \
    --eval-device-map cuda0 --map-dtype float16 --skip-anchor-eval

echo "QUEUE5_145A_DONE"
