#!/usr/bin/env bash
# Round 6 on .145, after three fixes:
#   - fp16 clamp bounds now round INWARD (the invariance assertion caught 130
#     escapes at 4B caused by outward fp16 rounding of the bound itself)
#   - the bnb anchor self-check tolerance was ~2% too tight at 8B
#   - the QIL fill is materialized in bf16, which is what OOMed 27B and 7B
# Memory rules learned the hard way: fp32 dense healing does not fit above
# ~4B on an 80 GB card, so 7B+ runs use the no-heal A path or QIL.
#   $1 = gpu index
#   systemd-run --user --unit clora-q6a --collect bash -c \
#     "cd ~/clora && bash experiments/run_queue6_145.sh 0 > out/q6a.log 2>&1"
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/hf-cache" HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${1:-0}"
PY=.venv/bin/python

run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $(date +%H:%M) $name ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

if [ "${1:-0}" = "0" ]; then
  # scale ladder, cheap end first
  run exp20_qil_0p6   exp5_qil.py       --model Qwen/Qwen3-0.6B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3
  run exp20_aplus_0p6 exp0_clip_rate.py --model Qwen/Qwen3-0.6B-Base --n-facts 1000 --epochs 24 --replay-frac 0.1 --heal-epochs 4 --heal-lr 2e-5
  run exp21_qil_4b    exp5_qil.py       --model Qwen/Qwen3-4B-Base   --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 --bs 8
  run exp21_aplus_4b  exp0_clip_rate.py --model Qwen/Qwen3-4B-Base   --n-facts 1000 --epochs 24 --replay-frac 0.1 --heal-epochs 4 --heal-lr 2e-5 --heal-optim adam8bit --bs 8
  run exp24_qil_8b    exp5_qil.py       --model Qwen/Qwen3-8B-Base   --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 --bs 8
  run exp24_a_8b      exp0_clip_rate.py --model Qwen/Qwen3-8B-Base   --n-facts 1000 --epochs 24 --replay-frac 0.1 --bs 8
else
  # family generality: no dense healing at 7B (fp32 heal does not fit)
  run exp23_mistral_a   exp0_clip_rate.py --model mistralai/Mistral-7B-v0.3 --n-facts 1000 --epochs 24 --replay-frac 0.1 --bs 8
  run exp23_mistral_qil exp5_qil.py       --model mistralai/Mistral-7B-v0.3 --n-facts 1000 --epochs 24 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 --bs 8
  # QIL at 27B, now that the fill is bf16
  run exp13_qil_27b     exp5_qil.py --model Qwen/Qwen3.8-27B --n-facts 1000 --epochs 8 --replay-frac 0.1 --rank 64 --tanh-scale 40 --lr 1e-3 --bs 4 --margin 0.02 --map-dtype float16 --skip-anchor-eval --eval-dtype bfloat16
fi

echo "QUEUE6_DONE_gpu${1:-0}"
