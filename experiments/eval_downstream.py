#!/usr/bin/env python
"""Downstream benchmarks for the served model.

The paper prices knowledge against perplexity, on the argument that LAMBADA is
held out from rehearsal. A referee will still ask whether the model that
absorbed the facts is a good model, and perplexity is a weak proxy for that.
This runs the standard multiple-choice suite through lm-eval-harness on the
served weights.

Three things are worth being careful about.

  * The served object is `W_hat + delta` in fp16 -- a plain dense model, not a
    4-bit one -- so the comparison that matters is against the *anchor*
    (dequantized 4-bit, no update), not against the original fp32 release.
    Both are evaluated here so the reader can see the quantization cost and the
    update cost separately.
  * lm-eval must see the same tokenizer and the same dtype in every arm, or
    the differences are dtype noise. We load once per arm and overwrite the
    weights in place.
  * At 1.7B most of this suite sits near chance, MMLU especially. A flat MMLU
    is evidence of nothing; the arms that discriminate at this scale are
    ARC-easy, HellaSwag and WinoGrande. We report all of them and say so.

Run:
  python experiments/eval_downstream.py --merged out/exp10_merged.pt \
      --tasks arc_easy,hellaswag,piqa,winogrande --out out/bench_r64.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp0_clip_rate import load_dense  # noqa: E402

DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--merged", default=None,
                   help="weight map from --save-merged-weights; omit to "
                        "evaluate the base model as released")
    p.add_argument("--anchor", default=None,
                   help="anchors-only weight map, i.e. the dequantized 4-bit "
                        "release with no update -- the correct baseline")
    p.add_argument("--fill", default=None,
                   help="factors from --save-fill: the served model is the "
                        "4-bit release with the fills re-attached (no dense "
                        "copy; the form that fits at 31B)")
    p.add_argument("--quantize-anchor", action="store_true",
                   help="build the anchor arm from the model itself: NF4 "
                        "quantize as the experiments do, dequantize to dense. "
                        "No weight map needed. For a published 4-bit release "
                        "the plain --model load already is the anchor.")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--chat-template", action="store_true",
                   help="score through the model's chat template (how an "
                        "instruction-tuned model is actually used); default "
                        "is raw completion scoring")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N docs per task (smoke test)")
    p.add_argument("--batch-size", default="8")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16")
    p.add_argument("--out", default="out/bench.json")
    return p.parse_args()


def overwrite(model, weight_map):
    """Swap the wrapped matrices in place; everything else stays as released."""
    n = 0
    for name, w in weight_map.items():
        p = model.get_parameter(f"{name}.weight")
        p.data.copy_(w.to(p.device, p.dtype))
        n += 1
    return n


def main():
    args = parse()
    t0 = time.time()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    dtype = getattr(torch, args.dtype)
    if args.fill:
        from experiments.served import load_served

        model, tok, label = load_served(args.model, fill=args.fill,
                                        dtype=dtype)
    elif args.quantize_anchor:
        from experiments.exp0_clip_rate import build_4bit, dequantize_in_place

        model, tok = build_4bit(args.model)
        n = dequantize_in_place(model)
        # transformers refuses .to(dtype) on a model it still flags as
        # bitsandbytes-quantized, although every Linear4bit is dense by
        # now (this arm failed at 27B); cast the parameters directly
        for q in model.parameters():
            q.data = q.data.to(dtype)
        print(f"[anchor] quantized {args.model} to NF4 and dequantized "
              f"{n} layers: this arm is the anchor", flush=True)
        label = "anchor (quantized in place)"
    else:
        model = load_dense(args.model, {"": 0}, dtype)
        tok = AutoTokenizer.from_pretrained(args.model)
        label = "released fp32"
    for path, name in ((args.anchor, "anchor"), (args.merged, "merged")):
        if path:
            n = overwrite(model, torch.load(path, map_location="cpu",
                                            weights_only=False))
            label = name
            print(f"[weights] {name}: overwrote {n} matrices", flush=True)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[eval] {label}: {tasks}", flush=True)
    res = simple_evaluate(model=lm, tasks=tasks, limit=args.limit,
                          apply_chat_template=args.chat_template,
                          bootstrap_iters=0)

    scores = {}
    for task, row in res["results"].items():
        for k, v in row.items():
            if k.startswith("acc") and not k.endswith("_stderr") \
                    and isinstance(v, (int, float)):
                scores.setdefault(task, {})[k] = round(float(v), 4)
    out = dict(model=args.model, arm=label, dtype=args.dtype,
               merged=args.merged, anchor=args.anchor, fill=args.fill,
               limit=args.limit,
               tasks=tasks, scores=scores,
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(scores, indent=2))
    print(f"[done] {args.out}  ({out['minutes']} min)")


if __name__ == "__main__":
    main()
