#!/usr/bin/env python
"""Knowledge fusion inside the cells: K writers, one released artifact.

Can several parties inject knowledge into copies of the same release,
independently and at the same time, and have the results added up? With
fills bounded by tanh the arithmetic says yes, if each party is given 1/K of
the half-width: every fill satisfies |f_k| <= M/K, so the sum satisfies
|sum f_k| <= M and the aggregate is bitwise-invariant *by construction* --
no clamp, nothing lost at the merge, and each party's contribution remains
separately revocable. This measures whether the knowledge survives the
addition the way the arithmetic does.

Two ways to train the parties, three ways to merge them, same facts and
budget throughout:

  --training reserved   each party trains with fill_frac = 1/K, i.e. it
                        knows in advance it will be diluted and its tanh can
                        saturate to compensate.  Merge: SUM (invariant by
                        construction, |sum| <= M).
  --training full       each party trains at full width, independently, with
                        no knowledge of K.  Two merges from the same parties:
                          average  (1/K) * sum -- invariant by convexity: the
                                   cell is an interval, and a mean of points
                                   inside it stays inside it. No coordination,
                                   no clamp.
                          clamp    sum, then clamp to the cell -- what a reader
                                   would try first; loses information wherever
                                   two parties push one weight the same way.
  K = 1 with either setting is the single-writer reference (the archived r64
  headline recipe).

The layer partition showed that the loss at the merge is not arithmetic:
parties writing DISJOINT matrices still lose half their recall when summed
(K=2: 35/39% in place, 23/15% fused). What one party's fill does to another
party's inputs is a functional perturbation, f_k x_j, and it lands on facts
whose logit margins are small. So the algorithm has to make each fill inert
where it was not written, or resolve the collision per weight:

  post-hoc rules on full-width parties (--how; all produce |t| <= 1 per
  coordinate and are therefore invariant by construction, no clamp):
    magnitude   per weight, the party with the larger |t| wins outright --
                no dilution, conflicts settled by who cares more
    ties        TIES-Merging (Yadav et al. 2023) in t-space: trim each party
                to its top --ties-keep fraction by |t|, elect the sign per
                weight, mean of the agreeing parties
    dare        DARE (Yu et al. 2024): drop a random --dare-drop fraction of
                each party, rescale the rest by 1/(1-p), sum, clamp
  writers that are inert elsewhere (training-time):
    --inert W        add W * KL(p_base || p_fill) on a neutral pool every step
                     (the released model with fills switched off is the
                     teacher): the fill may change the served distribution on
                     its own facts and nowhere else.  --inert-pool wikitext
                     uses generic text; neutral uses fact-shaped sentences
                     about entities no party was given (synthetic only), i.e.
                     the shared schema without anyone's content.
    --orthogonal R   OSRM-style (Zhang & Zhou, ACL 2025): before training,
                     each fill's input factor A is projected off the top-R
                     principal directions of the layer inputs produced by the
                     other parties' texts (--orth-pool others) or by generic
                     text (--orth-pool wikitext), then frozen; to first order
                     the fill then cannot respond to those inputs.
    --freeze-a       the control for --orthogonal: A frozen at random, no
                     projection.
  --save-parties / --merge-only let every rule be scored on the same trained
  parties.

The question the averaging arm asks is whether knowledge survives being
diluted K-fold at merge time; the reserved arm asks whether a party that
trains under the dilution encodes it more robustly. Each party's in-place
recall on its own facts is recorded before merging, so a loss after merging
is attributable to the merge and not to training.

  .venv/bin/python experiments/exp_fuse.py --parties 2 --training reserved
  .venv/bin/python experiments/exp_fuse.py --parties 2 --training full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    COMPUTE_DTYPE,
    build_4bit, eval_fp32_variants, eval_ppl, eval_recall, lambada_text,
    maybe_ppl, train, wikitext_text, wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill, wrap_model  # noqa: E402
from experiments.synth_facts import (  # noqa: E402
    generate, probe_pairs, training_texts,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--n-facts", type=int, default=1000)
    p.add_argument("--facts-file", default=None,
                   help="a real corpus (JSON list of {text, domain, ...}); "
                        "parties are then whole domains -- the hospital "
                        "writes medicine, the newsroom writes the rest -- "
                        "which is the situation fusion is for")
    p.add_argument("--probes-file", default=None)
    p.add_argument("--domain-split", default=None,
                   help="comma-separated groups of domains, one per party, "
                        "e.g. 'medicine|lottery,technology,sports,...'; "
                        "default: medicine alone against everything else")
    p.add_argument("--parties", type=int, default=2)
    p.add_argument("--training", choices=["reserved", "full", "partition"],
                   default="reserved",
                   help="partition: parties write disjoint SETS OF MATRICES "
                        "at full width (party k owns the layers whose index "
                        "is k mod K); the merge is a plain sum with no "
                        "shared weight, so nothing dilutes and nothing "
                        "collides -- the PackNet-style allocation of the "
                        "cell budget")
    p.add_argument("--how", default=None,
                   help="comma list of merge rules; default by --training: "
                        "sum for reserved/partition, average,clamp for full")
    p.add_argument("--ties-keep", type=float, default=0.2)
    p.add_argument("--dare-drop", type=float, default=0.5)
    p.add_argument("--inert", type=float, default=0.0,
                   help="weight of the KL-to-base term on the neutral pool")
    p.add_argument("--inert-pool", choices=["wikitext", "neutral"],
                   default="wikitext")
    p.add_argument("--inert-bs", type=int, default=8)
    p.add_argument("--orthogonal", type=int, default=0,
                   help="project A off the top-R input directions of the "
                        "--orth-pool and freeze it (0 = off)")
    p.add_argument("--orth-pool", choices=["others", "wikitext"],
                   default="others")
    p.add_argument("--freeze-a", action="store_true")
    p.add_argument("--save-parties", default=None,
                   help="write the trained factors, probes and in-place "
                        "records here, so other rules can be scored later")
    p.add_argument("--merge-only", default=None,
                   help="skip training; merge the parties saved by "
                        "--save-parties with the rules in --how")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--out", default="out/exp_fuse.json")
    return p.parse_args()


def fills_of(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, BoundedFill)]


def owner_of(name: str, K: int) -> int:
    """Which party owns this matrix under the layer partition."""
    import re

    m = re.search(r"layers\.(\d+)\.", name)
    return (int(m.group(1)) % K) if m else 0


def reset_party(model, seed, frac, party=None, K=1):
    """Fresh low-rank factors for a new writer; B=0 so the fill starts at 0.

    Under the partition, matrices this party does not own are frozen at zero
    fill: their factors get no gradient, and they contribute nothing to the
    merge, so each weight has exactly one writer at full width.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, m in fills_of(model):
        r, in_f = m.A.shape
        m.A.data.copy_(torch.randn(r, in_f, generator=g) / math.sqrt(in_f))
        m.B.data.zero_()
        m.fill_frac = frac
        owned = party is None or owner_of(name, K) == party
        m.A.requires_grad_(owned)
        m.B.requires_grad_(owned)


def snapshot(model):
    return {n: (m.A.detach().clone().cpu(), m.B.detach().clone().cpu())
            for n, m in fills_of(model)}


def combine(ts, how, K, frac, rule_args, stats, seed=0):
    """One merge rule, in t-space (each t_k in (-1, 1)); returns t in [-1, 1].

    sum        reserved parties: frac * sum, |.| <= K * frac = 1
    average    mean -- convex, so inside the cell without a clamp
    clamp      sum then clamp: loses wherever two parties push the same way
    magnitude  per weight, the party with the largest |t| -- a selection,
               hence inside the cell
    ties       trim each party to its top `ties_keep` fraction by |t|, elect
               the sign per weight by the sum of the kept values, mean of
               the kept values that agree -- a mean of in-cell points
    dare       drop each kept value with probability `dare_drop`, rescale
               by 1/(1-p), sum, clamp
    """
    if how == "sum":
        return frac * torch.stack(ts).sum(0)
    if how == "average":
        return torch.stack(ts).mean(0)
    if how == "clamp":
        t = torch.stack(ts).sum(0)
        stats["n_clamped"] = stats.get("n_clamped", 0) + int((t.abs() > 1).sum().item())
        return t.clamp(-1.0, 1.0)
    if how == "magnitude":
        st = torch.stack(ts)
        idx = st.abs().argmax(0, keepdim=True)
        return st.gather(0, idx)[0]
    if how == "ties":
        keep = rule_args["ties_keep"]
        trimmed = []
        for t in ts:
            n_keep = max(1, int(keep * t.numel()))
            thr = t.abs().kthvalue(t.numel() - n_keep + 1).values
            trimmed.append(torch.where(t.abs() >= thr, t,
                                       torch.zeros_like(t)))
        st = torch.stack(trimmed)
        elected = torch.sign(st.sum(0))
        agree = (torch.sign(st) == elected) & (st != 0)
        num = (st * agree).sum(0)
        den = agree.sum(0).clamp(min=1)
        stats["n_conflict"] = stats.get("n_conflict", 0) + int(
            ((st != 0).sum(0) > agree.sum(0)).sum().item())
        return num / den
    if how == "dare":
        p_drop = rule_args["dare_drop"]
        g = torch.Generator(device=ts[0].device).manual_seed(seed)
        t = torch.zeros_like(ts[0])
        for tk in ts:
            keep = (torch.rand(tk.shape, generator=g, device=tk.device)
                    >= p_drop)
            t += tk * keep / (1 - p_drop)
        stats["n_clamped"] = stats.get("n_clamped", 0) + int((t.abs() > 1).sum().item())
        return t.clamp(-1.0, 1.0)
    raise ValueError(how)


def merge_parties(model, frozen, parties, margin, how, map_dtype,
                  rule_args=None):
    """Combine the parties' fills inside each cell and verify invariance.

    how: one of the rules in `combine`; every rule keeps |t| <= 1, so the
    result is inside the cell by construction and the check below is a
    check, not a repair.
    """
    rule_args = rule_args or dict(ties_keep=0.2, dare_drop=0.5)
    merged, anchors_map, stats = {}, {}, dict(n_bad=0, n_sat=0, n_clamped=0,
                                              n_tot=0)
    K = len(parties)
    with torch.no_grad():
        for name, mod in fills_of(model):
            codes, absmax, bsz, anchors = frozen[name]
            dev = mod.A.device
            codes_d, absmax_d = codes.to(dev), absmax.to(dev)
            lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                                margin=margin)
            anch = anchors.to(dev).reshape(-1).float()
            m_exact = torch.minimum(anch - lo, hi - anch)
            ts = []
            for party in parties:
                A, B = party[name]
                ts.append(torch.tanh(mod.tanh_scale
                                     * (B.to(dev) @ A.to(dev)).float()
                                     ).reshape(-1))
            t_total = combine(ts, how, K, mod.fill_frac, rule_args, stats,
                              seed=zlib.crc32(name.encode()) & 0xFFFF)
            w_new = anch + m_exact * t_total
            _, n_mis, _ = check_invariance(w_new, codes_d, absmax_d, bsz)
            stats["n_bad"] += n_mis
            stats["n_sat"] += int((t_total.abs() > 0.99).sum().item())
            stats["n_tot"] += t_total.numel()
            merged[name] = w_new.reshape(anchors.shape).to(map_dtype).cpu()
            anchors_map[name] = anchors.to(map_dtype)
    if stats["n_bad"]:
        raise RuntimeError(
            f"structural invariance violated on {stats['n_bad']} weights")
    return merged, anchors_map, stats


_CUR_MASK = {}


@torch.no_grad()
def input_covariances(model, tok, texts, bs=16):
    """E[x x^T] of the input of every fill, over the non-pad tokens of texts."""
    covs, hooks = {}, []

    def make(name):
        def pre(mod, inputs):
            x = inputs[0].detach()
            mask = _CUR_MASK["m"]
            x = x.reshape(-1, x.shape[-1])[mask.reshape(-1).bool()].float()
            c = covs.get(name)
            covs[name] = x.T @ x if c is None else c + x.T @ x
        return pre

    for name, m in fills_of(model):
        hooks.append(m.register_forward_pre_hook(make(name)))
    model.eval()
    tok.padding_side = "right"
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=96).to(model.device)
        _CUR_MASK["m"] = enc.attention_mask
        model(**enc)
    for h in hooks:
        h.remove()
    return covs


@torch.no_grad()
def orthogonalize(model, covs, r):
    """A <- A (I - U U^T), U the top-r eigenvectors of the input covariance;
    then A is frozen so the constraint holds throughout training."""
    removed = []
    for name, m in fills_of(model):
        c = covs[name]
        evals, evecs = torch.linalg.eigh(c)
        U = evecs[:, -r:]
        A = m.A.data.float()
        before = A.norm()
        A = A - (A @ U) @ U.T
        m.A.data.copy_(A.to(m.A.dtype))
        m.A.requires_grad_(False)
        removed.append(1 - (A.norm() / before).item() ** 2)
        frac_var = (evals[-r:].sum() / evals.sum().clamp(min=1e-12)).item()
        covs[name] = None
        removed[-1] = (removed[-1], frac_var)
    return removed


@torch.no_grad()
def crosstalk(model, tok, parties, pairs, K, n_per=150, bs=24):
    """How much each writer's fill moves the served distribution on every
    writer's prompts: mean KL(p_base || p_base+f_k) over the next-token
    distribution at the end of the prompt, party k's fill alone attached.
    The diagonal is what a writer meant to do; the off-diagonal is the
    functional interference a merge inherits."""
    from experiments.exp0_clip_rate import fills_off

    prompts = {}
    for q, _, tag in pairs:
        prompts.setdefault(tag, []).append(q)
    prompts = {t: v[:n_per] for t, v in sorted(prompts.items())}
    tok.padding_side = "left"
    mods = dict(fills_of(model))

    def last_logprobs():
        out = {}
        for t, ps in prompts.items():
            rows = []
            for i in range(0, len(ps), bs):
                enc = tok(ps[i:i + bs], return_tensors="pt",
                          padding=True).to(model.device)
                lg = model(**enc).logits[:, -1].float()
                rows.append(torch.log_softmax(lg, -1))
            out[t] = torch.cat(rows)
        return out

    with fills_off(model):
        base = last_logprobs()
    table = {}
    for k, party in enumerate(parties):
        for name, m in mods.items():
            A, B = party[name]
            m.A.copy_(A.to(m.A.device))
            m.B.copy_(B.to(m.B.device))
        lp = last_logprobs()
        table[f"party{k}"] = {
            t: float((base[t].exp() * (base[t] - lp[t])).sum(-1).mean())
            for t in prompts}
    return table


def main():
    args = parse()
    t0 = time.time()
    K = args.parties
    frac = 1.0 / K if args.training == "reserved" else 1.0
    partition = args.training == "partition"
    torch.manual_seed(args.seed)

    if args.facts_file:
        rows = json.loads(Path(args.facts_file).read_text())
        probe_path = Path(args.probes_file or
                          args.facts_file.replace("_train", "_probes"))
        probes = [q for q in json.loads(probe_path.read_text())
                  if q.get("kind") != "twohop"]
        doms = sorted({r.get("domain", "real") for r in rows})
        if args.domain_split:
            groups = [g.split("|") for g in args.domain_split.split(",")]
        else:
            groups = [["medicine"], [d for d in doms if d != "medicine"]]
        assert len(groups) == K, f"{len(groups)} domain groups for K={K}"
        parts = [[r for r in rows if r.get("domain", "real") in g]
                 for g in groups]
        texts_of = lambda fs: [r["text"] for r in fs]  # noqa: E731
        pairs = [(q["prompt"], q["answer"], f"party{k}")
                 for k, g in enumerate(groups) for q in probes
                 if q.get("domain", "real") in g]
        print(f"[data] real corpus: parties by domain {groups}; "
              f"{[len(x) for x in parts]} sentences, {len(pairs)} probes",
              flush=True)
    else:
        facts = generate(args.n_facts, seed=args.seed)
        parts = [facts[k::K] for k in range(K)]
        texts_of = training_texts
        # probes tagged by party so the per-party recall falls out of
        # eval_recall
        pairs = [(q, a, f"party{k}") for k, fs in enumerate(parts)
                 for q, a, _ in probe_pairs(fs)]

    model, tok = build_4bit(args.model)
    ppl_txt, x_txt = wikitext_text(), lambada_text()
    frozen = wrap_model(model, args.rank, args.tanh_scale, args.margin)

    kl_pool = None
    if args.inert > 0:
        if args.inert_pool == "neutral" and not args.facts_file:
            # entities no party was given: names drawn from the same space
            # collide by chance (about 1%), and those are dropped
            given = {f.name for fs in parts for f in fs}
            kl_pool = training_texts([f for f in generate(args.n_facts,
                                                          seed=args.seed + 999)
                                      if f.name not in given])
        else:
            kl_pool = wikitext_train_snippets(4000, seed=args.seed + 7)
        print(f"[inert] KL weight {args.inert} on {len(kl_pool)} "
              f"{args.inert_pool} texts", flush=True)

    parties, inplace = [], []
    if args.merge_only:
        saved = torch.load(args.merge_only, map_location="cpu",
                           weights_only=False)
        parties, inplace, pairs = saved["parties"], saved["inplace"], \
            saved["pairs"]
        frac = saved["frac"]
        K = len(parties)
        for _, m in fills_of(model):
            m.fill_frac = frac
        print(f"[merge-only] {len(parties)} parties from {args.merge_only}",
              flush=True)
        parts = [None] * len(parties)
    for k, fs in enumerate(parts):
        if args.merge_only:
            break
        texts = texts_of(fs)
        n_rep = 0
        pool = None
        if args.replay_frac > 0:
            n_rep = int(len(texts) * args.replay_frac / (1 - args.replay_frac))
            pool = wikitext_train_snippets(n_rep * args.epochs,
                                           seed=args.seed * 100 + k)
        reset_party(model, args.seed * 100 + k, frac,
                    party=(k if partition else None), K=K)
        orth = None
        if args.orthogonal > 0:
            if args.orth_pool == "others":
                o_texts = [t for kk, f2 in enumerate(parts) if kk != k
                           for t in texts_of(f2)]
            else:
                o_texts = wikitext_train_snippets(2000, seed=args.seed + 11)
            covs = input_covariances(model, tok, o_texts)
            orth = orthogonalize(model, covs, args.orthogonal)
            del covs
            torch.cuda.empty_cache()
            print(f"[party {k}] A projected off top-{args.orthogonal} "
                  f"directions of {len(o_texts)} {args.orth_pool} texts; "
                  f"norm removed {sum(a for a, _ in orth) / len(orth):.3f}, "
                  f"variance covered {sum(b for _, b in orth) / len(orth):.3f}",
                  flush=True)
        elif args.freeze_a:
            for _, m in fills_of(model):
                m.A.requires_grad_(False)
        print(f"[party {k}] {len(fs)} facts, {len(texts)} sentences, "
              f"fill_frac={frac:.3f}", flush=True)
        losses = train(model, tok, texts, args.epochs, args.lr, args.bs,
                       args.seed * 100 + k, replay_pool=pool,
                       n_replay_per_epoch=n_rep, kl_pool=kl_pool,
                       kl_weight=args.inert, kl_bs=args.inert_bs)
        longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                      for _, a, _ in pairs)
        rec, by = eval_recall(model, tok, pairs, detail=True,
                              max_new=max(32, longest + 4))
        own = by[f"party{k}"]
        others = {kk: v for kk, v in by.items() if kk != f"party{k}"}
        print(f"[party {k}] in-place recall own={own:.3f} others={others}",
              flush=True)
        inplace.append(dict(party=k, n_facts=len(fs), recall_own=own,
                            recall_by_party=by,
                            loss_last=sum(losses[-10:]) / 10,
                            orthogonal=orth))
        parties.append(snapshot(model))

    if args.save_parties and not args.merge_only:
        Path(args.save_parties).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(parties=parties, inplace=inplace, pairs=pairs,
                        frac=frac, config=vars(args)), args.save_parties)
        print(f"[saved] parties -> {args.save_parties}", flush=True)

    xt = crosstalk(model, tok, parties, pairs, K)
    for k, row in xt.items():
        print(f"[crosstalk] {k}: " + "  ".join(f"{t}={v:.3f}" for t, v in row.items()),
              flush=True)

    if args.how:
        hows = args.how.split(",")
    else:
        hows = (["sum"] if args.training in ("reserved", "partition")
                else ["average", "clamp"])
    rule_args = dict(ties_keep=args.ties_keep, dare_drop=args.dare_drop)

    # a published 4-bit release dequantizes to bf16 layers; the dense
    # evaluation copy has to be loaded in the same dtype (the synthetic runs
    # start from a dense release and are evaluated in fp32 as archived)
    from transformers import AutoConfig

    published = getattr(AutoConfig.from_pretrained(args.model),
                        "quantization_config", None)
    eval_dtype = COMPUTE_DTYPE if published else torch.float32
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    arms = {}
    # one merge at a time, evaluated before the next is built: five dense
    # fp32 maps of a 1.7B model held together are 28 GB of host RAM, and a
    # K=4 run was killed by the kernel between its second and third rule
    for how in hows:
        print(f"[merge] {how} over {K} parties", flush=True)
        merged, anchors_map, mstats = merge_parties(
            model, frozen, parties, args.margin, how, torch.float16,
            rule_args=rule_args)
        print(f"[merge] invariance ok; "
              f"saturation={mstats['n_sat'] / mstats['n_tot']:.4%}  "
              f"clamped={mstats['n_clamped']}  "
              f"conflicts={mstats.get('n_conflict', 0)}", flush=True)
        fp32 = eval_fp32_variants(args.model, anchors_map, merged, tok, pairs,
                                  ppl_txt, args.max_ppl_chunks, xdom_txt=x_txt,
                                  max_new=max(32, longest + 4),
                                  dtype=eval_dtype)
        del merged, anchors_map
        print(f"[fused:{how}] recall={fp32['recall_merged']:.4f} by party="
              f"{fp32['recall_merged_by_kind']}  ppl={fp32['ppl_merged']:.3f}"
              f"  lambada={fp32['ppl_x_merged']}", flush=True)
        arms[how] = dict(
            merge=dict(invariance_violations=0,
                       saturation=mstats["n_sat"] / mstats["n_tot"],
                       clamped=mstats["n_clamped"],
                       conflicts=mstats.get("n_conflict", 0),
                       n_weights=mstats["n_tot"]),
            recall=dict(fused=fp32["recall_merged"],
                        fused_by_party=fp32["recall_merged_by_kind"],
                        anchor=fp32["recall_anchor_fp32"],
                        original=fp32["recall_original_fp32"]),
            ppl=dict(fused=fp32["ppl_merged"],
                     anchor=fp32["ppl_anchor_fp32"],
                     original=fp32["ppl_original_fp32"]),
            ppl_lambada=dict(fused=fp32["ppl_x_merged"],
                             anchor=fp32["ppl_x_anchor_fp32"],
                             original=fp32["ppl_x_original_fp32"]))

    out = dict(config=vars(args), training=args.training, parties=inplace,
               crosstalk=xt, arms=arms,
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  ({out['minutes']} min)")


if __name__ == "__main__":
    main()
