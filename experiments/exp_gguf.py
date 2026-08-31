#!/usr/bin/env python
"""In-cell learning on a GGUF (llama.cpp) release: the file people run.

The GGUF is read twice. transformers dequantizes it into a dense model for
the forward pass; cellfill.gguf_grid unpacks the same file into per-weight
codes, scales and offsets, so that every matrix's cells -- width |s|,
centred on the anchor, clamped at the code range -- are defined against
the shipped blocks. The dense weights of each matrix are replaced by the
exact anchors (the two agree to bf16 rounding, which is checked), a
BoundedFill with room |s|(1/2 - margin) is wrapped around it, and the
fill is trained as everywhere else. After training the served weights are
re-binned under the shipped scales and offsets and must return the stored
codes on every weight; the fill is saved; the file was never touched.

Q4_K_M mixes Q4_K and Q6_K matrices; both are written. Embeddings and the
output head are left alone.

  python experiments/exp_gguf.py --repo unsloth/Qwen3-1.7B-GGUF \
      --file Qwen3-1.7B-Q4_K_M.gguf --facts-file data/allaug_train.json \
      --probes-file data/allplus_probes.json --out out/exp82_gguf_1p7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill.gguf_grid import cells, codes_changed, read_gguf, unpack  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    COMPUTE_DTYPE,
    eval_ppl, eval_recall, lambada_text, maybe_ppl, train, wikitext_text,
    wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill, _probe_counts  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--facts-file", required=True)
    p.add_argument("--probes-file", required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--save-fill", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    torch.manual_seed(args.seed)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_gguf_pytorch_utils import (
        TENSOR_PROCESSORS, TensorProcessor, get_gguf_hf_weights_map,
    )

    path = hf_hub_download(args.repo, args.file)
    tok = AutoTokenizer.from_pretrained(args.repo, gguf_file=args.file)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, gguf_file=args.file, dtype=COMPUTE_DTYPE,
        device_map={"": 0})
    model.eval()
    print(f"[load] {args.file} dequantized by transformers: "
          f"{type(model).__name__}", flush=True)

    # the cells of every quantized matrix, by HF module name
    reader, tensors = read_gguf(path)
    arch = model.config.model_type
    proc = TENSOR_PROCESSORS.get(arch, TensorProcessor)(config=model.config.to_dict())
    name_map = get_gguf_hf_weights_map(model, proc)
    mods = dict(model.named_modules())
    grids, types, n_written, n_frozen_tensors = {}, {}, 0, 0
    max_dev = 0.0
    for gname, (qt, blocks, shape) in tensors.items():
        hf = name_map.get(gname)
        if hf is None or not hf.endswith(".weight"):
            continue
        mod = mods.get(hf[:-len(".weight")])
        if not isinstance(mod, nn.Linear) or "lm_head" in hf:
            continue
        if blocks is None:
            n_frozen_tensors += 1
            continue
        q, s, o, qmin, qmax = unpack(blocks, qt)
        group = 16 if qt == "Q6_K" else 32
        anchor, lo, hi, room = cells(q, s, o, qmin, qmax, args.margin)
        anchor = anchor.view(shape)
        dev = mod.weight.device
        # the dense copy transformers made must be these anchors up to bf16
        # rounding (relative 2^-8; the 8B has entries of magnitude 30)
        wcpu = mod.weight.detach().float().cpu()
        rel = ((wcpu - anchor).abs() / (anchor.abs() + 1e-3)).max().item()
        max_dev = max(max_dev, rel)
        with torch.no_grad():
            mod.weight.copy_(anchor.to(mod.weight.dtype).to(dev))
        # scales and offsets are constant within a block: keep one per block
        grids[hf[:-len(".weight")]] = dict(
            codes=torch.as_tensor(q, dtype=torch.int16),
            s_g=torch.as_tensor(s[::group]).float(),
            o_g=torch.as_tensor(o[::group]).float(),
            room_g=room[::group].float(), group=group,
            qmin=qmin, qmax=qmax, shape=shape)
        types[qt] = types.get(qt, 0) + 1
        n_written += anchor.numel()
    print(f"[cells] {len(grids)} matrices {types}, {n_written:,} weights; "
          f"dense copy vs anchors max relative diff {max_dev:.3e} (bf16 rounding)",
          flush=True)
    assert max_dev < 1e-2, "transformers' dequantization disagrees with the cells"

    # wrap
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for name, g in grids.items():
        parent_name, child = name.rsplit(".", 1)
        parent = mods[parent_name]
        base = getattr(parent, child)
        fill = BoundedFill(base, None, args.rank, args.tanh_scale,
                           margin=args.margin, groom=(g["room_g"], g["group"]))
        fill.checkpoint_fill = True
        setattr(parent, child, fill)
    fills = {n: m for n, m in model.named_modules() if isinstance(m, BoundedFill)}
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[wrap] {len(fills)} fills, trainable params {n_train:,}", flush=True)

    rows = json.loads(Path(args.facts_file).read_text())
    texts = [r["text"] for r in rows]
    probes = json.loads(Path(args.probes_file).read_text())
    pairs = [(q_["prompt"], q_["answer"], q_.get("kind", q_.get("domain", "?")))
             for q_ in probes if q_.get("kind") != "twohop"]
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    max_new = max(32, longest + 4)
    ppl_txt, x_txt = wikitext_text(), lambada_text()
    rec0, by0 = eval_recall(model, tok, pairs, detail=True, max_new=max_new)
    ppl0 = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
    lam0 = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
    print(f"[released] recall={rec0:.4f} ppl={ppl0:.3f} lambada={lam0}", flush=True)

    n_rep = int(len(texts) * args.replay_frac / (1 - args.replay_frac))
    pool = wikitext_train_snippets(n_rep * args.epochs, seed=args.seed)
    train(model, tok, texts, args.epochs, args.lr, args.bs, args.seed,
          replay_pool=pool, n_replay_per_epoch=n_rep)
    model.eval()
    rec, by = eval_recall(model, tok, pairs, detail=True, max_new=max_new)
    ppl = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
    lam = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
    print(f"[trained] recall={rec:.4f} ppl={ppl:.3f} lambada={lam}  by kind: "
          + "  ".join(f"{k}={v:.1%}" for k, v in sorted(by.items())), flush=True)

    if args.save_fill:
        torch.save(dict(meta=dict(model=f"{args.repo}/{args.file}", rank=args.rank,
                                  tanh_scale=args.tanh_scale, margin=args.margin,
                                  grid="gguf", fill_frac=1.0, link="tanh"),
                        fills={n: (m.A.detach().cpu(), m.B.detach().cpu())
                               for n, m in fills.items()}), args.save_fill)
        print(f"[save] fill -> {args.save_fill}", flush=True)

    # the check: re-bin the served weights under the shipped scales and offsets
    n_bad = n_tot = n_sat = 0
    with torch.no_grad():
        for name, m in fills.items():
            g = grids[name]
            dev = m.A.device
            n = g["shape"][0] * g["shape"][1]
            s_w = g["s_g"].to(dev).repeat_interleave(g["group"])[:n]
            o_w = g["o_g"].to(dev).repeat_interleave(g["group"])[:n]
            codes = g["codes"].to(dev).float()
            anchor = s_w * codes + o_w                      # exact, fp32
            t = m.t_value(torch.float32).reshape(-1)
            room = m.halfwidth(torch.float32).reshape(-1)
            w = anchor + room * t
            bad = codes_changed(w, codes, s_w, o_w, g["qmin"], g["qmax"])
            n_bad += bad
            n_tot += w.numel()
            n_sat += int((t.abs() > 0.99).sum().item())
    print(f"[verify] {n_bad} of {n_tot:,} codes would change; saturation "
          f"{n_sat / max(n_tot, 1):.2%}", flush=True)
    res = dict(config=vars(args), grid_types=types, n_weights=n_tot,
               probe_counts=_probe_counts(pairs),
               recall=dict(released=rec0, released_by_kind=by0,
                           merged=rec, merged_by_kind=by, trained_inplace=rec),
               ppl=dict(released=ppl0, merged=ppl, trained_inplace=ppl),
               ppl_lambada=dict(released=lam0, merged=lam, trained_inplace=lam),
               merge=dict(invariance_violations=n_bad,
                          saturation=n_sat / max(n_tot, 1)),
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] {args.out}  ({res['minutes']} min)")
    if n_bad:
        raise SystemExit(f"structural invariance violated on {n_bad} weights")


if __name__ == "__main__":
    main()
