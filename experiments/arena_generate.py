#!/usr/bin/env python
"""Answer the Arena-Hard-Auto prompts, keeping everything a harness discards.

The question this file exists to answer is not "how good are the answers"
but "where does the score leak on the way out of the model". A post-trained
model with a thinking mode emits a reasoning span before its answer. A
harness that decodes with `skip_special_tokens=True` and scores the whole
string is then scoring the reasoning as if it were the answer, and the
model loses points it had already earned. We saw exactly this on
Qwen3-32B, whose completion-style MBPP fell to 14.6% -- below a 1.7B base
model -- for no reason but the prefix.

So this script records the raw decode with special tokens intact, the
answer after the thinking span is removed, the number of new tokens, and
whether generation stopped because the model finished or because it ran
out of budget. Every later claim about "format loss" is a comparison
between two of those fields, and can be recomputed from this file without
a GPU.

  python experiments/arena_generate.py \
      --model /home/zssy/models/Qwen3.8-27B \
      --questions data/arena_hard_v01.jsonl \
      --out out/arena_gen_27b.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A thinking span as Qwen writes it. Kept as one pattern so the stripping
# rule is stated once and reused by the classifier.
THINK = re.compile(r"<think>.*?</think>", re.S)
THINK_OPEN = re.compile(r"<think>")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--questions", required=True,
                   help="jsonl with a 'uid' and a 'prompt' field")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=4096,
                   help="Arena-Hard's own default; a thinking model needs "
                        "room for the span as well as the answer")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="first N prompts only, for a smoke run")
    p.add_argument("--dtype", choices=["bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--load-4bit", action="store_true",
                   help="load through bitsandbytes NF4 instead of dense; "
                        "needed to fit the model on a 48 GB card")
    p.add_argument("--enable-thinking", choices=["default", "on", "off"],
                   default="default",
                   help="passed to the chat template when it accepts it; "
                        "'off' is the ablation that measures what the "
                        "thinking span costs")
    # Qwen3.8's card gives different sampling for the two modes and warns
    # that the model is tuned for them. Greedy was the first thing tried
    # here, for reproducibility, and it is the wrong default: decoding a
    # model at an operating point it was not tuned for produces degenerate
    # output that reads as a model failure and is really a harness failure
    # -- the exact confusion this whole study exists to separate. Variance
    # is controlled with --seed and repeat runs instead.
    p.add_argument("--sampling", choices=["thinking", "instruct", "greedy"],
                   default="thinking",
                   help="thinking: t=1.0 top_p=0.95 top_k=20 (card default); "
                        "instruct: t=0.7 top_p=0.80 top_k=20 "
                        "presence_penalty=1.5; greedy: deterministic, "
                        "off-spec for this model, for debugging only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None,
                   choices=[None, "xhigh", "medium", "low"],
                   help="passed to the chat template when it accepts it; "
                        "the card's default is xhigh")
    return p.parse_args()


# Straight from the model card's recommended sets.
SAMPLING = {
    "thinking": dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20),
    "instruct": dict(do_sample=True, temperature=0.7, top_p=0.80, top_k=20,
                     repetition_penalty=1.0, presence_penalty=1.5),
    "greedy": dict(do_sample=False),
}


def load(model_id: str, dtype: str, four_bit: bool):
    import transformers

    kwargs = {"device_map": "auto"}
    major = int(transformers.__version__.split(".")[0])
    kwargs["dtype" if major >= 5 else "torch_dtype"] = getattr(torch, dtype)
    if four_bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype))
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    # Load with CausalLM and let a genuine class mismatch -- and only that --
    # fall back. A swallowed OSError once cost us a day of misdiagnosis.
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs)
    except (ValueError, KeyError) as e:
        print(f"[load] CausalLM refused the config ({type(e).__name__}); "
              f"trying ImageTextToText", flush=True)
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, **kwargs)
    model.eval()
    return model, tok


def render(tok, prompt: str, thinking: str, effort: str | None) -> str:
    """Apply the chat template, honouring the switches the card documents.

    Both switches are optional and template-specific, so each is tried and
    dropped if the template will not take it -- silently falling back to the
    template's own default is correct here, but it has to be announced,
    because a run that quietly ignored --enable-thinking off would look like
    evidence that the thinking span costs nothing.
    """
    msgs = [{"role": "user", "content": prompt}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if thinking != "default":
        kw["enable_thinking"] = (thinking == "on")
    if effort:
        kw["reasoning_effort"] = effort
    try:
        return tok.apply_chat_template(msgs, **kw)
    except TypeError as e:
        print(f"[render] template rejected a switch ({e}); retrying with "
              f"the template's defaults -- the ablation is NOT in effect",
              flush=True)
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def strip_thinking(text: str) -> str:
    """The answer as a reader would see it: closed spans removed, and an
    unclosed span treated as swallowing everything after it, because that
    is what it does to the reader too."""
    out = THINK.sub("", text)
    m = THINK_OPEN.search(out)
    if m:
        out = out[:m.start()]
    return out.strip()


def main():
    a = parse()
    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[arena] {len(rows)} prompts from {a.questions}", flush=True)

    torch.manual_seed(a.seed)
    model, tok = load(a.model, a.dtype, a.load_4bit)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        # Resume: a 27B pass over 500 prompts is long enough that losing it
        # to one bad batch is not acceptable.
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
        print(f"[arena] resuming, {len(done)} already written", flush=True)

    todo = [r for r in rows if r["uid"] not in done]
    t0 = time.time()
    written = 0
    with open(out_path, "a") as fh:
        for i in range(0, len(todo), a.bs):
            batch = todo[i:i + a.bs]
            texts = [render(tok, r["prompt"], a.enable_thinking,
                            a.reasoning_effort) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new,
                                     pad_token_id=tok.pad_token_id,
                                     **SAMPLING[a.sampling])
            new = gen[:, enc.input_ids.shape[1]:]
            for r, seq in zip(batch, new):
                # Trim right padding before counting, or every row reports
                # the length of the longest row in its batch.
                keep = seq
                if tok.pad_token_id is not None:
                    nz = (seq != tok.pad_token_id).nonzero()
                    if len(nz):
                        keep = seq[:nz[-1].item() + 1]
                n_new = int(keep.shape[0])
                raw = tok.decode(keep, skip_special_tokens=False)
                clean = strip_thinking(
                    tok.decode(keep, skip_special_tokens=True))
                rec = {
                    "uid": r["uid"],
                    "cluster": r.get("cluster"),
                    "prompt": r["prompt"],
                    "raw": raw,
                    "answer": clean,
                    "n_new_tokens": n_new,
                    # Budget-exhausted rather than finished: the single most
                    # common way a thinking model scores zero on a task it
                    # could do.
                    "truncated": n_new >= a.max_new,
                    "thinking_mode": a.enable_thinking,
                    "sampling": a.sampling,
                    "reasoning_effort": a.reasoning_effort,
                    "seed": a.seed,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            el = time.time() - t0
            print(f"[arena] {written}/{len(todo)}  {el/60:.1f} min", flush=True)

    print(f"[arena] wrote {written} rows to {out_path} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
