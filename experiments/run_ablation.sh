#!/usr/bin/env bash
# Tier-1 method-ablation queue. Waits for any running experiment, then runs
# each config serially on the single GPU. Idempotent: skips runs whose JSON
# already exists. Usage (on the server):
#   nohup bash experiments/run_ablation.sh > out/ablation.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com
PY=.venv/bin/python

while pgrep -f "exp0_[c]lip" >/dev/null; do sleep 30; done

run() {
  name=$1; shift
  if [ -f "out/$name.json" ]; then echo "skip $name (exists)"; return; fi
  echo "=== $name: $* ==="
  $PY experiments/exp0_clip_rate.py "$@" --out "out/$name.json" \
      > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# A+: QLoRA -> clip-merge -> projected dense heal (fresh replay throughout)
run exp2_heal  --n-facts 1000 --epochs 24 --replay-frac 0.3 --heal-epochs 4 --heal-lr 2e-5
# B: projected dense from anchors, no LoRA (constraint-aware from step 0)
run exp3_dense --n-facts 1000 --epochs 24 --replay-frac 0.3 --dense-only --heal-lr 2e-5
# Upper bound: unconstrained dense full FT (no box; violations reported)
run exp4_free  --n-facts 1000 --epochs 24 --replay-frac 0.3 --dense-only --no-clamp --heal-lr 2e-5

# QIL-LoRA: invariance by construction (bounded tanh fill)
if [ ! -f out/exp5_qil.json ]; then
  echo "=== exp5_qil ==="
  $PY experiments/exp5_qil.py --n-facts 1000 --epochs 24 --replay-frac 0.3 \
      --out out/exp5_qil.json > out/exp5_qil.log 2>&1
  echo "=== exp5_qil exit $? ==="
fi

echo "ABLATION_QUEUE_DONE"
