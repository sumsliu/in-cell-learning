#!/usr/bin/env python
"""How many bits did the fill store? Measured, not inferred from recall.

Exact-match recall is a threshold on the argmax and swings by ten points
between seeds at the same fact count. The quantity a capacity law should
be written in is information: for every attribute of every fact, the
model's log-probability of the true value, renormalized over the
attribute's vocabulary, gives the residual uncertainty H(a | model), and
the bits stored for that attribute are

    log2 |V_a| - H(a | model)    (clipped at 0)

summed over attributes and facts. The prior is uniform over the generator's
vocabulary, which is what the generator draws from, so this is the
bit-complexity measure of Allen-Zhu & Li's knowledge-capacity laws, taken
on cloze prompts. The three probe attributes (city 20, job 16, company 12:
11.9 bits) are scored on the recall probes; birth month (12) and year (75)
on the training sentence's prefix. Bits per trainable parameter and per
written cell follow from the wrapped model.

  python experiments/eval_bits.py --model Qwen/Qwen3-1.7B-Base --n-facts 10000 \
      --fill out/fill_cap10000_r128.pt --out out/bits_cap10000_r128.json
  python experiments/eval_bits.py ... --ckpt out/exp73_30000.ckpt --rank 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.served import continuation_logprob, load_served  # noqa: E402
from experiments.synth_facts import (  # noqa: E402
    CITIES, COMPANIES, JOBS, MONTHS, YEAR_HI, YEAR_LO, generate,
)

YEARS = [str(y) for y in range(YEAR_LO, YEAR_HI + 1)]
ATTRS = {
    "city": (lambda f: (f"{f.name} grew up in the city of", f.city), CITIES),
    "job": (lambda f: (f"{f.name} now works as a", f.job), JOBS),
    "company": (lambda f: (f"{f.name} is employed at", f.company), COMPANIES),
    "month": (lambda f: (f"{f.name} was born in", f.birth_month), MONTHS),
    "year": (lambda f: (f"{f.name} was born in {f.birth_month}",
                        str(f.birth_year)), YEARS),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-facts", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fill", default=None)
    p.add_argument("--ckpt", default=None,
                   help="a training checkpoint (params A/B by name); needs "
                        "--rank/--tanh-scale to wrap the model identically")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--fill-frac", type=float, default=1.0)
    p.add_argument("--link", default="tanh")
    p.add_argument("--sample", type=int, default=2000,
                   help="facts scored (seeded subsample); 0 = all")
    p.add_argument("--attrs", default="city,job,company,month,year")
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load(args):
    if args.fill:
        model, tok, label = load_served(args.model, None, fill=args.fill)
    else:
        from experiments.exp0_clip_rate import build_4bit
        from experiments.exp5_qil import BoundedFill, wrap_model

        model, tok = build_4bit(args.model)
        wrap_model(model, args.rank, args.tanh_scale, args.margin)
        mods = {n: m for n, m in model.named_modules()
                if isinstance(m, BoundedFill)}
        for m in mods.values():
            m.fill_frac, m.link_fn = args.fill_frac, args.link
        label = "released"
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            by_name = dict(model.named_parameters())
            for n, v in ck["params"].items():
                by_name[n].data.copy_(v.to(by_name[n].device, by_name[n].dtype))
            label = f"{Path(args.ckpt).stem}@epoch{ck['epoch']}"
            print(f"[ckpt] {label}: {len(ck['params'])} factors", flush=True)
        model.eval()
    from experiments.exp5_qil import BoundedFill

    fills = [m for m in model.modules() if isinstance(m, BoundedFill)]
    n_param = sum(m.A.numel() + m.B.numel() for m in fills)
    n_cells = sum(m.numel for m in fills)
    return model, tok, label, n_param, n_cells


@torch.no_grad()
def attribute_bits(model, tok, facts, attr, bs):
    """Per fact: bits stored about `attr` = log2|V| - H(a|model), H taken
    over the vocabulary; also whether the true value is the argmax."""
    mk, vocab = ATTRS[attr]
    prompts, conts, owner = [], [], []
    for i, f in enumerate(facts):
        pr, _ = mk(f)
        for v in vocab:
            prompts.append(pr)
            conts.append(" " + v)
            owner.append(i)
    lp = continuation_logprob(model, tok, prompts, conts, bs=bs)
    V = len(vocab)
    lp = torch.tensor(lp).view(len(facts), V)
    logz = torch.logsumexp(lp, dim=1, keepdim=True)
    post = lp - logz                          # log p(v | prompt, v in V)
    truth = torch.tensor([vocab.index(mk(f)[1]) for f in facts])
    h = -post.gather(1, truth[:, None])[:, 0] / math.log(2)   # bits
    stored = (math.log2(V) - h).clamp(min=0)
    top1 = (post.argmax(1) == truth).float()
    return stored, top1, h


def main():
    args = parse()
    t0 = time.time()
    facts = generate(args.n_facts, seed=args.seed)
    if args.sample and args.sample < len(facts):
        rng = random.Random(args.seed + 17)
        facts = rng.sample(facts, args.sample)
    model, tok, label, n_param, n_cells = load(args)
    attrs = args.attrs.split(",")
    res = dict(config=vars(args), label=label, n_scored=len(facts),
               n_trainable=n_param, n_cells=n_cells, attrs={})
    # The released model is scored on the same facts with the fills switched
    # off: its non-uniform priors (it prefers some cities) register as a
    # fraction of a bit per attribute even with nothing stored, and the
    # capacity law is written in the excess over that floor.
    from experiments.exp0_clip_rate import fills_off

    total_stored = total_prior = total_floor = 0.0
    for a in attrs:
        stored, top1, h = attribute_bits(model, tok, facts, a, args.bs)
        with fills_off(model):
            floor, top1_0, _ = attribute_bits(model, tok, facts, a, args.bs)
        V = len(ATTRS[a][1])
        res["attrs"][a] = dict(vocab=V, prior_bits=math.log2(V),
                               stored_bits_mean=stored.mean().item(),
                               released_bits_mean=floor.mean().item(),
                               residual_bits_mean=h.mean().item(),
                               top1=top1.mean().item(),
                               top1_released=top1_0.mean().item(),
                               stored_frac=(stored.mean() / math.log2(V)).item())
        total_stored += stored.mean().item()
        total_floor += floor.mean().item()
        total_prior += math.log2(V)
        print(f"[{a:8s}] |V|={V:3d} prior {math.log2(V):.2f} b  stored "
              f"{stored.mean():.3f} b/fact ({stored.mean() / math.log2(V):.1%})"
              f"  released {floor.mean():.3f}  top1 {top1.mean():.3f} "
              f"(released {top1_0.mean():.3f})", flush=True)
    excess = max(total_stored - total_floor, 0.0)
    res.update(stored_bits_per_fact=total_stored, released_bits_per_fact=total_floor,
               excess_bits_per_fact=excess, prior_bits_per_fact=total_prior,
               stored_frac=total_stored / total_prior,
               excess_frac=excess / total_prior,
               stored_bits_total=excess * args.n_facts,
               bits_per_trainable_param=excess * args.n_facts / n_param,
               bits_per_cell=excess * args.n_facts / n_cells,
               minutes=round((time.time() - t0) / 60, 1))
    print(f"[bits] {total_stored:.2f} of {total_prior:.2f} bits per fact "
          f"({res['stored_frac']:.1%}), released floor {total_floor:.2f}; "
          f"excess {excess:.2f} b/fact, total {res['stored_bits_total']:.3e} "
          f"bits = {res['bits_per_trainable_param']:.3f} per trainable param, "
          f"{res['bits_per_cell']:.2e} per cell", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
