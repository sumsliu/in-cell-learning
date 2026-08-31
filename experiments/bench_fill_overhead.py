#!/usr/bin/env python
"""What does carrying a fill cost at inference time?

The deployment form this project argues for is the released 4-bit artifact
kept exactly as shipped, with the update re-attached as a separate low-rank
file. That form is the whole point -- the vendor's file never changes -- but
its cost has so far only been estimated on paper: roughly 0.7% more storage
and 1.5% more multiply-accumulates for GLM-4.5-Air.

Two things that estimate does not capture. The bound M is rebuilt from the
frozen codes on every forward rather than stored, so its cost is arithmetic
the FLOP count does not see. And decoding is memory-bound, where an extra
elementwise pass over a weight-shaped tensor can matter more than its share
of the FLOPs suggests.

So this measures it instead: the same model, the same prompts, the same
decode length, through the same loader, differing only in whether the fill
is attached.

  python experiments/bench_fill_overhead.py --model /path/to/model \
      --fill out/fill_27b.pt --new-tokens 128 --reps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain the difference between a process and a thread.",
    "Given a list of integers, return the indices of the two that sum to a target.",
    "Summarise what a B-tree is and when a database would choose one.",
]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", required=True, help="the fill to attach for arm B")
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--reps", type=int, default=3,
                   help="timed repetitions after one discarded warm-up")
    p.add_argument("--bs", type=int, default=1,
                   help="1 keeps this in the memory-bound decode regime, "
                        "which is where the bound rebuild is most likely to "
                        "show up")
    p.add_argument("--out", default="out/fill_overhead.json")
    return p.parse_args()


def measure(model, tok, prompts, new_tokens, reps, bs):
    """Prefill and decode timings, plus the peak memory the arm needed."""
    tok.padding_side = "left"
    enc = tok(prompts[:bs], return_tensors="pt", padding=True).to(model.device)
    torch.cuda.reset_peak_memory_stats()

    def one(n_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, out.shape[1] - enc.input_ids.shape[1]

    one(4)  # warm-up: CUDA graphs, autotune, allocator
    # A single new token is almost all prefill; subtracting it from the full
    # run separates the two phases without needing hooks.
    pre = min(one(1)[0] for _ in range(reps))
    full, n_new = min((one(new_tokens) for _ in range(reps)),
                      key=lambda r: r[0])
    dec = full - pre
    return {
        "prefill_s": round(pre, 4),
        "decode_s": round(dec, 4),
        "n_new_tokens": int(n_new),
        "decode_tok_s": round((n_new - 1) / dec, 2) if dec > 0 else None,
        "total_s": round(full, 4),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def main():
    a = parse()
    from experiments.served import load_served

    results = {}
    for arm, fill in (("base_4bit", None), ("base_4bit_plus_fill", a.fill)):
        print(f"[bench] loading {arm}", flush=True)
        model, tok, label = load_served(a.model, fill=fill)
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        results[arm] = measure(model, tok, PROMPTS, a.new_tokens, a.reps, a.bs)
        results[arm]["label"] = label
        print(f"  {arm}: {json.dumps(results[arm])}", flush=True)
        del model
        torch.cuda.empty_cache()

    b, f = results["base_4bit"], results["base_4bit_plus_fill"]
    if b["decode_tok_s"] and f["decode_tok_s"]:
        results["overhead"] = {
            "decode_slowdown_pct": round(
                (b["decode_tok_s"] / f["decode_tok_s"] - 1) * 100, 2),
            "prefill_slowdown_pct": round(
                (f["prefill_s"] / b["prefill_s"] - 1) * 100, 2),
            "extra_peak_mem_gb": round(f["peak_mem_gb"] - b["peak_mem_gb"], 2),
        }
    results["config"] = {"model": a.model, "fill": a.fill, "bs": a.bs,
                         "new_tokens": a.new_tokens, "reps": a.reps}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results.get("overhead", {}), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
