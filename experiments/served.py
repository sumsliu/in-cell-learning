"""Load a served model: a release, optionally with saved matrices swapped in.

Every evaluation that runs after training (downstream suites, the two-hop
ceiling, the API usage test) needs the same object: the model as it would be
served, which is the dense release with the wrapped matrices overwritten by
`--save-merged-weights` output, or by the anchors-only map for the control.
This is that loader, so the evaluators cannot drift from one another in how
they build the thing they measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp0_clip_rate import load_dense  # noqa: E402


def overwrite(model, weight_map) -> int:
    """Swap the wrapped matrices in place; everything else stays as released."""
    n = 0
    for name, w in weight_map.items():
        p = model.get_parameter(f"{name}.weight")
        p.data.copy_(w.to(p.device, p.dtype))
        n += 1
    return n


def load_served(model_id: str, weights: str | None = None,
                dtype=None, device_map=None,
                fill: str | None = None):
    """The released model, with an update applied if given.

    Two forms of the same served function:
      weights  a dense map from --save-merged-weights, written over a dense
               copy of the release (a published 4-bit release is dequantized
               first);
      fill     the low-rank factors from --save-fill, re-attached to the
               release kept in its 4-bit form. Nothing dense is built, so
               this is the form that fits at 31B.
    Returns (model, tok, label), label naming the arm for the record.
    """
    from transformers import AutoConfig, AutoTokenizer

    if dtype is None:
        from experiments.exp0_clip_rate import COMPUTE_DTYPE

        dtype = COMPUTE_DTYPE
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    published = getattr(AutoConfig.from_pretrained(model_id),
                        "quantization_config", None)
    if fill:
        from experiments.exp0_clip_rate import build_4bit
        from experiments.exp5_qil import BoundedFill, wrap_model

        ck = torch.load(fill, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        model, tok = build_4bit(model_id)
        wrap_model(model, meta["rank"], meta["tanh_scale"], meta["margin"],
                   name_filter=meta.get("target_filter"),
                   codebook_m=meta.get("codebook_m", False))
        mods = {n: m for n, m in model.named_modules()
                if isinstance(m, BoundedFill)}
        if set(mods) != set(ck["fills"]):
            raise RuntimeError("fill file does not match the wrapped layers: "
                               f"{len(mods)} wrapped, {len(ck['fills'])} saved")
        with torch.no_grad():
            for n, (A, B) in ck["fills"].items():
                mods[n].A.copy_(A.to(mods[n].A.device))
                mods[n].B.copy_(B.to(mods[n].B.device))
                # the width share and the link the fill was trained with
                mods[n].fill_frac = meta.get("fill_frac", 1.0)
                mods[n].link_fn = meta.get("link", "tanh")
                if meta.get("stack", 1) > 1:
                    mods[n].add_stack(meta["stack"], meta.get("stack_mode", "drf"))
                    for q, v in zip(mods[n].stack, ck["stacks"][n]):
                        q.copy_(v.to(q.device))
        label = Path(fill).stem
        print(f"[fill] {label}: re-attached {len(mods)} fills", flush=True)
        model.eval()
        return model, tok, label
    model = load_dense(model_id, device_map or {"": 0}, dtype)
    label = "released-4bit" if published else "released"
    if weights:
        n = overwrite(model, torch.load(weights, map_location="cpu",
                                        weights_only=False))
        label = Path(weights).stem
        print(f"[weights] {label}: overwrote {n} matrices", flush=True)
    model.eval()
    return model, tok, label


def generate_batch(model, tok, prompts, max_new=48, bs=16):
    """Greedy continuations, left-padded, special tokens stripped."""
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt",
                  padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True)
    tok.padding_side = "right"
    return outs


def continuation_logprob(model, tok, prompts, continuations, bs=16):
    """Sum of token log-probabilities of each continuation given its prompt.

    Teacher-forced, right-padded: the prompt tokens are masked out of the
    sum, so two continuations of different length under the same prompt are
    compared on their own tokens only. Used to rank candidate completions,
    which asks a base model what it knows without asking it to follow an
    instruction.
    """
    import torch.nn.functional as F

    tok.padding_side = "right"
    out = []
    for i in range(0, len(prompts), bs):
        ps, cs = prompts[i:i + bs], continuations[i:i + bs]
        p_ids = [tok(p, add_special_tokens=True).input_ids for p in ps]
        c_ids = [tok(c, add_special_tokens=False).input_ids for c in cs]
        seqs = [a + b for a, b in zip(p_ids, c_ids)]
        L = max(len(x) for x in seqs)
        pad = tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x)
            att[r, :len(x)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for r, (a, b) in enumerate(zip(p_ids, c_ids)):
            start = len(a) - 1                     # position predicting c[0]
            out.append(float(tok_lp[r, start:start + len(b)].sum()))
    return out
