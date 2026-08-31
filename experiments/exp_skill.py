#!/usr/bin/env python
"""A skill patch: does a bounded fill carry an ability, not only facts?

Everything above writes facts. A vendor patch or a vertical model is an
ability---write code, follow a format---and whether that fits inside the
cells is a different question, because an ability is not a list of
associations but a set of directions the fine-tuning literature finds to
be low-rank and small. This trains the same bounded fill on code
instructions (Magicoder-OSS-Instruct, Python), response-only loss, on the
vendor's 4-bit release, and scores pass@1 on HumanEval and MBPP against
the released model and against QLoRA with the same rank on the same
release. Drift is read on WikiText and LAMBADA. For the fill, the merged
weights re-quantize to the released codes (checked); for LoRA the number
of codes its merge would change is counted, which is the difference
between a patch and a new file.

  python experiments/exp_skill.py --model unsloth/Qwen3-1.7B-Base-bnb-4bit \
      --method cellfill --n-train 10000 --epochs 2 --out out/skill_cellfill_1p7b.json
  python experiments/exp_skill.py --model ... --method lora ...
  python experiments/exp_skill.py --model ... --method none ...   (the release)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    build_4bit, eval_ppl, lambada_text, maybe_ppl, wikitext_text,
    wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill, materialize, wrap_model  # noqa: E402

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

FMT = "### Problem:\n{problem}\n\n### Solution:\n"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=["cellfill", "lora", "none"],
                   required=True)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-3 for cellfill, 2e-4 for lora")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--link", default="tanh")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default="humaneval,mbpp")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-bs", type=int, default=8)
    p.add_argument("--save-fill", default=None)
    p.add_argument("--codebook-m", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def code_examples(n, seed):
    from datasets import load_dataset

    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    rows = [r for r in ds if r["lang"] == "python"]
    random.Random(seed).shuffle(rows)
    return [(FMT.format(problem=r["problem"].strip()), r["solution"].strip())
            for r in rows[:n]]


def encode(tok, prompt, response, max_length):
    """prompt tokens masked out of the loss; truncated from the end."""
    p_ids = tok(prompt, add_special_tokens=True).input_ids
    r_ids = tok(response + tok.eos_token, add_special_tokens=False).input_ids
    ids = (p_ids + r_ids)[:max_length]
    labels = ([-100] * len(p_ids) + r_ids)[:max_length]
    return ids, labels


def train(model, tok, pairs, epochs, lr, bs, max_length, seed, replay):
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    model.train()
    losses, step = [], 0
    n_rep = int(len(pairs) * replay / (1 - replay)) if replay > 0 else 0
    pool = wikitext_train_snippets(n_rep * epochs, seed=seed) if n_rep else []
    for ep in range(epochs):
        items = list(pairs)
        k0 = (ep * n_rep) % max(len(pool), 1)
        items += [(None, pool[(k0 + j) % len(pool)]) for j in range(n_rep)]
        rng.shuffle(items)
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            enc = []
            for pr, rs in batch:
                if pr is None:          # rehearsal: plain LM loss
                    ids = tok(rs, add_special_tokens=True,
                              truncation=True, max_length=max_length).input_ids
                    enc.append((ids, list(ids)))
                else:
                    enc.append(encode(tok, pr, rs, max_length))
            L = max(len(a) for a, _ in enc)
            ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(enc), L), -100, dtype=torch.long)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for r, (a, b) in enumerate(enc):
                ids[r, :len(a)] = torch.tensor(a)
                lab[r, :len(b)] = torch.tensor(b)
                att[r, :len(a)] = 1
            dev = model.device
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                        labels=lab.to(dev))
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss))
            step += 1
            if step % 25 == 0:
                print(f"  ep{ep} step{step} loss {sum(losses[-25:]) / 25:.4f}",
                      flush=True)
    model.eval()
    return losses


def lm_eval_scores(model, tok, tasks, limit, bs):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit,
                          bootstrap_iters=0, confirm_run_unsafe_code=True)
    out = {}
    for t, r in res["results"].items():
        out[t] = {k: v for k, v in r.items()
                  if isinstance(v, (int, float)) and "stderr" not in k}
    return out


@torch.no_grad()
def lora_codes_changed(model, frozen, margin):
    """If the LoRA delta were merged into the dequantized release, how many
    4-bit codes would change under the frozen scales."""
    from peft.tuners.lora import Linear as LoraLinear

    n_bad = n_tot = 0
    mods = {n: m for n, m in model.named_modules() if isinstance(m, LoraLinear)}
    for name, (codes, absmax, bsz, anchors) in frozen.items():
        m = None
        for k, v in mods.items():
            if k.endswith(name) or name.endswith(k.replace("base_model.model.", "")):
                m = v
                break
        if m is None:
            continue
        A = m.lora_A["default"].weight.float()
        B = m.lora_B["default"].weight.float()
        delta = (B @ A) * m.scaling["default"]
        dev = delta.device
        w = anchors.to(dev).float().reshape(-1) + delta.reshape(-1)
        _, n, _ = check_invariance(w, codes.to(dev), absmax.to(dev), bsz)
        n_bad += n
        n_tot += w.numel()
    return n_bad, n_tot


def main():
    args = parse()
    t0 = time.time()
    torch.manual_seed(args.seed)
    tasks = [t for t in args.tasks.split(",") if t]
    model, tok = build_4bit(args.model)
    tok.padding_side = "right"
    ppl_txt, x_txt = wikitext_text(), lambada_text()
    res = dict(config=vars(args))
    frozen = None
    if args.method == "cellfill":
        frozen = wrap_model(model, args.rank, args.tanh_scale, args.margin,
                            codebook_m=args.codebook_m)
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.checkpoint_fill = True
                m.link_fn = args.link
        lr = args.lr or 1e-3
    elif args.method == "lora":
        from peft import LoraConfig, get_peft_model
        from cellfill.bnb_state import frozen_state_from_linear4bit

        frozen = {}
        for name, mod in model.named_modules():
            if type(mod).__name__ == "Linear4bit":
                fs = frozen_state_from_linear4bit(mod)
                frozen[name] = (fs["codes"].cpu(), fs["absmax"].cpu(),
                                fs["blocksize"], fs["anchors"].half().cpu())
        for q in model.parameters():
            q.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM"))
        for q in model.parameters():
            if q.requires_grad:
                q.data = q.data.float()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        lr = args.lr or 2e-4
    else:
        lr = None
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[{args.method}] trainable params {n_train:,}", flush=True)

    if args.method != "none":
        pairs = code_examples(args.n_train, args.seed)
        print(f"[data] {len(pairs)} python instruction pairs; max_length "
              f"{args.max_length}", flush=True)
        losses = train(model, tok, pairs, args.epochs, lr, args.bs,
                       args.max_length, args.seed, args.replay_frac)
        res["loss_first"] = sum(losses[:25]) / max(len(losses[:25]), 1)
        res["loss_last"] = sum(losses[-25:]) / max(len(losses[-25:]), 1)

    if args.method == "lora":
        model.gradient_checkpointing_disable()
    res["ppl"] = eval_ppl(model, tok, ppl_txt, max_chunks=40)
    res["ppl_lambada"] = maybe_ppl(model, tok, x_txt, 40)
    print(f"[drift] wikitext {res['ppl']:.3f}  lambada {res['ppl_lambada']}",
          flush=True)
    if args.method == "cellfill":
        if args.save_fill:
            torch.save(dict(meta=dict(model=args.model, rank=args.rank,
                                      tanh_scale=args.tanh_scale,
                                      margin=args.margin,
                                      codebook_m=args.codebook_m,
                                      target_filter=None, fill_frac=1.0,
                                      link=args.link),
                            fills={n: (m.A.detach().cpu(), m.B.detach().cpu())
                                   for n, m in model.named_modules()
                                   if isinstance(m, BoundedFill)}),
                       args.save_fill)
        _, _, sat = materialize(model, frozen, args.margin,
                                map_dtype=torch.float16, store_anchors=False,
                                skip_dense=True)
        res["invariance"] = dict(violations=0, saturation=sat)
        print(f"[invariance] 0 violations, saturation {sat:.2%}", flush=True)
    elif args.method == "lora":
        n_bad, n_tot = lora_codes_changed(model, frozen, args.margin)
        res["invariance"] = dict(codes_changed=n_bad, n_weights=n_tot,
                                 frac=n_bad / max(n_tot, 1))
        print(f"[invariance] LoRA merge would change {n_bad:,} of {n_tot:,} "
              f"codes ({n_bad / max(n_tot, 1):.2%})", flush=True)
    tok.padding_side = "left"
    res["scores"] = lm_eval_scores(model, tok, tasks, args.limit, args.eval_bs)
    print(f"[scores] {res['scores']}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
