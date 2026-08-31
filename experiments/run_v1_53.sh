#!/usr/bin/env bash
# v1 critical path on .53, in strict priority order.
#
# An adversarial audit (2026-08-20) found three headline claims in the paper
# that the repo's own JSONs contradict. These runs are what turn those into
# defensible statements; everything else in the round-5 queue is v2 material.
#
#   1. exp4 seeds  -- the "cost of invariance is 1.8 points" claim rests on
#      n=1 for the unconstrained arm, and one constrained seed (exp3_s2,
#      64.4%) already beats that "upper bound" (63.0%). 40 min settles it.
#   2. QIL r16 @ replay 0.1 -- the rank ablation moved rank AND replay
#      together (r16 runs are replay 0.3, r64 is 0.1), and the paper itself
#      attributes +7.7 points to that replay change.
#   3. 10k capacity with cross-domain eval -- the flagship 4130-facts number
#      has no LAMBADA at all, and at 1k the same method degrades LAMBADA by
#      ~150%. The capacity claim is currently uncheckable off-domain.
#   4. exp_seq -- the only experiment that supports the word "continual".
#
#   nohup bash experiments/run_v1_53.sh > out/v1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
PY=.venv/bin/python

# let whatever is on the GPU right now finish (exp_geom at time of writing)
while pgrep -f "exp_[g]eom" >/dev/null \
   || pgrep -f "exp0_[c]lip" >/dev/null \
   || pgrep -f "exp5_[q]il" >/dev/null \
   || pgrep -f "exp_[s]eq" >/dev/null; do sleep 30; done

run() {
  name=$1; script=$2; shift 2
  [ -f "out/$name.json" ] && { echo "skip $name"; return; }
  echo "=== $(date +%H:%M) $name: $* ==="
  $PY "experiments/$script" "$@" --out "out/$name.json" > "out/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# 1. unconstrained-baseline seeds (~21 min each)
for S in 1 2; do
  run "exp4_free_s$S" exp0_clip_rate.py --n-facts 1000 --epochs 24 \
      --replay-frac 0.3 --dense-only --no-clamp --heal-lr 2e-5 --seed "$S"
done

# 2. de-confound the QIL rank ablation: r16 at the SAME replay as r64 (~13 min)
run exp16_qil_r16_replay10 exp5_qil.py --n-facts 1000 --epochs 24 \
    --replay-frac 0.1 --rank 16 --tanh-scale 40 --lr 1e-3

# 3. capacity at 10k WITH cross-domain eval, weights archived for the codec
run exp17_b10k_xdom exp0_clip_rate.py --n-facts 10000 --epochs 24 \
    --replay-frac 0.3 --dense-only --heal-lr 2e-5 --probe-cap 12000 \
    --save-merged
run exp17_aplus10k_xdom exp0_clip_rate.py --n-facts 10000 --epochs 24 \
    --replay-frac 0.1 --rank 64 --alpha 128 --heal-epochs 4 --heal-lr 2e-5 \
    --probe-cap 12000

# 4. the continual-learning lifecycle
SEQ="--model Qwen/Qwen3-1.7B-Base --facts-per-task 500 --epochs 24 --rank 64 --replay-frac 0.1"
run exp15_seq_minor       exp_seq.py $SEQ --tasks 4 --mode minor
run exp15_seq_consolidate exp_seq.py $SEQ --tasks 4 --mode consolidate \
    --consolidate-after 2

echo "V1_53_DONE"
