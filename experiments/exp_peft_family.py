#!/usr/bin/env python
"""Where does the update live? One corpus, one budget, four PEFT families.

The comparison this project has published so far is against LoRA only. That
leaves the central question of the taxonomy unanswered, because LoRA is the
one PEFT method whose update lives in the same place in-cell learning's does
-- the weight matrix -- and the interesting contrasts are with methods whose
update lives somewhere else entirely.

Four methods, each the representative of a different answer:

  lora    a low-rank residual ON the weight. Mergeable, and merging is what
          changes the released codes.
  ia3     learned rescaling vectors applied to activations. An inserted
          structure; folding it back into weights again moves them.
  prefix  trainable key/value states prepended to attention. The weights are
          never touched at all; the update lives in the KV cache and costs
          context at every forward pass.
  ln      only the LayerNorm parameters are trained. These are typically NOT
          quantized in a 4-bit release, so this method also leaves the INT4
          codes untouched -- by a completely different route from in-cell
          learning, and one worth measuring rather than assuming away.

Everything else is held fixed: the same released 4-bit model, the same
corpus, the same epochs, batch size, seed and replay fraction. What differs
is only where the parameters being trained live.

  python experiments/exp_peft_family.py --peft prefix --model <release> \
      --facts-file data/real1_all_train.json --probes-file data/allaug_probes.json \
      --epochs 24 --lr 1e-3 --bs 16 --out out/peft_prefix.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--peft", required=True,
                   choices=["lora", "ia3", "prefix", "ln"])
    p.add_argument("--model", default="unsloth/Qwen3-1.7B-Base-bnb-4bit")
    p.add_argument("--facts-file", required=True)
    p.add_argument("--probes-file", required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--rank", type=int, default=64,
                   help="LoRA rank; also the number of virtual tokens for "
                        "prefix tuning, so the two are budget-comparable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--max-ppl-chunks", type=int, default=30)
    p.add_argument("--out", default="out/peft_family.json")
    return p.parse_args()


def install(model, kind, rank):
    """Attach one PEFT family and say where its parameters live."""
    import torch
    from peft import get_peft_model

    if kind == "lora":
        from peft import LoraConfig
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                         bias="none", task_type="CAUSAL_LM",
                         target_modules="all-linear")
        where = "a low-rank residual on the quantized weight matrices"
    elif kind == "ia3":
        from peft import IA3Config
        cfg = IA3Config(task_type="CAUSAL_LM")
        where = "rescaling vectors applied to activations, outside the weights"
    elif kind == "prefix":
        from peft import PrefixTuningConfig
        cfg = PrefixTuningConfig(task_type="CAUSAL_LM",
                                 num_virtual_tokens=rank)
        where = "key/value states prepended at attention, in the KV cache"
    elif kind == "ln":
        from peft import LNTuningConfig
        cfg = LNTuningConfig(task_type="CAUSAL_LM")
        where = "the LayerNorm parameters, which a 4-bit release leaves unquantized"
    else:
        raise ValueError(kind)
    wrapped = get_peft_model(model, cfg)
    # PEFT creates its new parameters in fp32 while a 4-bit load computes in
    # bf16. LoRA's factors meet the activations through a matmul that promotes;
    # IA3's rescaling vectors meet them through an elementwise multiply that
    # does not, and it raises. Casting the trainable parameters to the compute
    # dtype is what makes the four families comparable at all -- the
    # alternative is to compare one method in fp32 against three in bf16.
    from experiments.exp0_clip_rate import COMPUTE_DTYPE

    n_cast = 0
    for q in wrapped.parameters():
        if q.requires_grad and q.dtype is torch.float32:
            q.data = q.data.to(COMPUTE_DTYPE)
            n_cast += 1
    if n_cast:
        print(f"[peft] cast {n_cast} trainable tensors to {COMPUTE_DTYPE}",
              flush=True)
    return wrapped, where


def main():
    a = parse()
    t0 = time.time()
    import torch

    from experiments.exp0_clip_rate import (
        build_4bit, eval_ppl, eval_recall, lambada_text, train, wikitext_text,
    )

    model, tok = build_4bit(a.model)
    if a.peft == "ia3":
        # IA3 rescales the key and value projections with learned vectors, and
        # the rescaled tensors come back in a different dtype from the query
        # under the fused attention kernel, which raises. The eager path does
        # the same arithmetic without the fused kernel's dtype assertion.
        try:
            model.set_attn_implementation("eager")
            print("[peft] ia3: eager attention (the fused kernel asserts on "
                  "mixed dtypes)", flush=True)
        except Exception:
            model.config._attn_implementation = "eager"

    facts = json.loads(Path(a.facts_file).read_text())
    probes = json.loads(Path(a.probes_file).read_text())
    texts = [r["text"] if isinstance(r, dict) else r for r in facts]
    pairs = [(r["prompt"], r["answer"]) for r in probes]
    print(f"[data] {len(texts)} facts, {len(pairs)} probes", flush=True)

    ppl_txt = wikitext_text()
    lam_txt = lambada_text()
    anchor_ppl = eval_ppl(model, tok, ppl_txt, max_chunks=a.max_ppl_chunks)
    anchor_lam = eval_ppl(model, tok, lam_txt, max_chunks=a.max_ppl_chunks)
    print(f"[anchor] wikitext {anchor_ppl:.2f}  lambada {anchor_lam:.2f}",
          flush=True)

    model, where = install(model, a.peft, a.rank)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[peft] {a.peft}: {trainable:,} trainable of {total:,}", flush=True)
    print(f"[peft] the update lives in {where}", flush=True)

    n_rep = int(len(texts) * a.replay_frac / (1 - a.replay_frac))
    replay_pool = ppl_txt.split("\n\n") if n_rep else None
    losses = train(model, tok, texts, a.epochs, a.lr, a.bs, a.seed,
                   replay_pool=replay_pool, n_replay_per_epoch=n_rep)

    recall = eval_recall(model, tok, pairs)
    served_ppl = eval_ppl(model, tok, ppl_txt, max_chunks=a.max_ppl_chunks)
    served_lam = eval_ppl(model, tok, lam_txt, max_chunks=a.max_ppl_chunks)

    # Whether the released integer codes survive is a property of where the
    # update lives, not something to be discovered per run; recording the
    # reasoning next to the numbers keeps the table honest.
    codes_survive = {
        "lora": "only while unmerged; merging moves the weights and the codes",
        "ia3": "only while unmerged; folding the rescaling moves the weights",
        "prefix": "yes, the weights are never touched -- but the update is not "
                  "in the artifact at all, it is context at every forward pass",
        "ln": "yes; a 4-bit release leaves LayerNorm unquantized, so this "
              "trains parameters the codes do not cover",
    }[a.peft]

    out = dict(
        peft=a.peft, model=a.model, where=where,
        trainable_params=trainable, total_params=total,
        recall=recall,
        ppl=dict(anchor=anchor_ppl, served=served_ppl),
        ppl_lambada=dict(anchor=anchor_lam, served=served_lam),
        released_codes_survive=codes_survive,
        config=vars(a), final_loss=losses[-1] if losses else None,
        minutes=(time.time() - t0) / 60,
    )
    from experiments.exp0_clip_rate import eval_ppl, eval_recall, stamp_of
    out["scorer"] = stamp_of(eval_recall, eval_ppl)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {a.peft}: recall {recall:.1%}  wikitext "
          f"{anchor_ppl:.2f}->{served_ppl:.2f}  lambada "
          f"{anchor_lam:.2f}->{served_lam:.2f}  ({out['minutes']:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
