#!/usr/bin/env python
"""In-cell learning against retrieval: the same facts, in the weights or in
the prompt.

Retrieval-augmented generation keeps the model fixed and puts the knowledge
in the context; a wiki the model consults is the same idea with a human
index. Both preserve the artifact trivially. Their costs are where this
method claims an advantage, so they have to be measured on the same facts
and the same probes, with the same scorer:

  none     the released model, no context -- the floor, and on the PopQA
           tail the measure of how much of this the model already knew
  oracle   the sentences that state the answer, placed in the prompt: the
           ceiling of any retriever, and a measure of how well the model
           uses context it is handed
  bm25@k   a lexical retriever over the haystack, top k in the prompt
  dense@k  a dense retriever (BAAI/bge-small-en-v1.5) over the same haystack
  fill     the injected model, no context (CellFill, --fill)
  fill+bm25@k  both: the injected model with retrieval on top

The haystack is every sentence of the corpus -- the documents the fill was
trained on, so neither side sees a different text -- plus, with
--haystack wikitext103:N, N paragraphs of WikiText-103 as distractors, which
is what an index of anything real looks like. Two probe formats:

  the repository's corpora (data/all*_probes.json): prompt is the probe
  itself, scored by the exact-prefix rule used everywhere else
  PopQA (data/popqa_*_probes.json, experiments/build_popqa.py): the probe
  carries the question and its aliases; the prompt is the fixed four-shot
  QA format with the retrieved facts before the question, scored by PopQA's
  rule (an alias appears in the answer line; whole-word, aliases of at
  least three characters)

Recorded per arm: recall by kind, the retriever's hit rate (an answer alias
is in what was retrieved), prompt length in tokens, wall time per probe.

  python experiments/exp_rag.py --model unsloth/Qwen3-1.7B-Base-bnb-4bit \
      --corpus data/popqa_tail_train.json --probes data/popqa_tail_probes.json \
      --haystack wikitext103:200000 --dense --fill out/fill_popqa_1p7b.pt \
      --out out/rag_popqa_1p7b.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.served import generate_batch, load_served  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")


def toks(s: str):
    return TOKEN.findall(s.lower())


class BM25:
    """Okapi BM25 with an inverted index; terms in more than a tenth of the
    documents are not scored (they are stop words, and their postings are
    the whole haystack)."""

    def __init__(self, docs, k1=1.5, b=0.75, max_df=0.1):
        self.raw = docs
        self.k1, self.b = k1, b
        self.len = []
        post = defaultdict(list)
        for i, d in enumerate(docs):
            t = toks(d)
            self.len.append(len(t))
            for term, f in Counter(t).items():
                post[term].append((i, f))
        n = len(docs)
        self.avg = sum(self.len) / max(n, 1)
        self.post, self.idf = {}, {}
        for term, pl in post.items():
            if len(pl) > max_df * n and n > 100:
                continue
            self.post[term] = pl
            self.idf[term] = math.log(1 + (n - len(pl) + 0.5)
                                      / (len(pl) + 0.5))

    def top(self, query, k):
        scores = defaultdict(float)
        for t in set(toks(query)):
            pl = self.post.get(t)
            if not pl:
                continue
            idf = self.idf[t]
            for i, f in pl:
                scores[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        best = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [self.raw[i] for i, _ in best]


class Dense:
    """bge-small-en-v1.5: CLS pooling, normalized, cosine; the query carries
    the model's retrieval instruction, the passages do not."""

    INSTR = "Represent this sentence for searching relevant passages: "

    def __init__(self, docs, device, name="BAAI/bge-small-en-v1.5",
                 cache=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.raw = docs
        self.tok = AutoTokenizer.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device
        if cache and Path(cache).exists():
            self.emb = torch.load(cache, map_location=device)
            assert self.emb.shape[0] == len(docs)
        else:
            self.emb = self.encode(docs, max_len=128)
            if cache:
                torch.save(self.emb.cpu(), cache)

    def encode(self, texts, max_len=64, bs=512):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                e = self.tok(texts[i:i + bs], padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt"
                             ).to(self.device)
                h = self.enc(**e).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h, dim=-1))
        return torch.cat(out)

    def top_batch(self, queries, k):
        q = self.encode([self.INSTR + x for x in queries])
        sims = q @ self.emb.T
        idx = sims.topk(k, dim=-1).indices.tolist()
        return [[self.raw[i] for i in row] for row in idx]


def hit_prefix(gen: str, answer: str) -> bool:
    """The repository's exact-prefix rule, as everywhere else."""
    return gen.strip().lower().startswith(answer.strip().lower())


def hit_alias(gen: str, aliases) -> bool:
    """PopQA's rule, tightened to whole words: the answer line contains one
    of the aliases (those of at least three characters)."""
    line = gen.strip().split("\n")[0].lower()
    for a in aliases:
        a = a.strip().lower()
        if len(a) < 3:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", line):
            return True
    return False


def in_context(ctx, pr) -> bool:
    als = [a for a in pr.get("aliases", [pr["answer"]]) if len(a.strip()) >= 3]
    text = "\n".join(ctx).lower()
    return any(a.lower() in text for a in als)


def load_haystack(spec: str, cache_dir="out"):
    """wikitext103:N -> N paragraphs of at least 200 characters from the
    WikiText-103 training split, cached as JSON next to the results."""
    kind, n = spec.split(":")
    n = int(n)
    cache = Path(cache_dir) / f"haystack_{kind}_{n}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    assert kind == "wikitext103", spec
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train")
    out = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) >= 200 and not t.startswith("="):
            out.append(t)
            if len(out) >= n:
                break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", default=None, help="factors from --save-fill")
    p.add_argument("--corpus", default="data/allaug_train.json")
    p.add_argument("--probes", default="data/allplus_probes.json")
    p.add_argument("--haystack", default=None,
                   help="distractor passages, e.g. wikitext103:200000")
    p.add_argument("--dense", action="store_true",
                   help="add the dense retriever arms")
    p.add_argument("--chunked", action="store_true",
                   help="index passages, not sentences: each corpus sentence "
                        "sits inside a WikiText-103 paragraph (the fact in "
                        "the middle of a page about something else), which "
                        "is what a chunked document index retrieves")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--oracle-n", type=int, default=4)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--base-arms", default=None,
                   help="comma list of arms to run on the released model "
                        "(default: all); the fill always runs none and bm25")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    import torch

    rows = json.loads(Path(args.corpus).read_text())
    sents = [r["text"] for r in rows]
    probes = json.loads(Path(args.probes).read_text())
    popqa = "question" in probes[0]
    ks = [int(k) for k in args.ks.split(",")]
    hay = list(sents)
    n_distract = 0
    if args.haystack:
        extra = load_haystack(args.haystack)
        n_distract = len(extra)
        if args.chunked:
            # the n-th sentence goes into the middle of the n-th paragraph
            # of the second half of the haystack; the first half stays as
            # distractors, so no chunk holds two facts
            host = extra[len(extra) // 2:]
            assert len(host) >= len(sents), "haystack too small to host"
            wrapped = []
            for i, s_ in enumerate(sents):
                para = host[i].split(" ")
                h = len(para) // 2
                wrapped.append(" ".join(para[:h]) + " " + s_ + " "
                               + " ".join(para[h:]))
            hay = wrapped + extra[:len(extra) // 2] + host[len(sents):]
            sents_doc = wrapped      # the documents that state the facts
        else:
            hay += extra
    if not (args.haystack and args.chunked):
        sents_doc = sents
    print(f"[haystack] {len(sents)} corpus sentences + {n_distract} "
          f"distractors", flush=True)
    t0 = time.time()
    bm = BM25(hay)
    print(f"[bm25] indexed {len(hay)} docs in {time.time() - t0:.0f}s",
          flush=True)

    if popqa:
        from experiments.build_popqa import prompt_of

        def build_prompt(ctx, pr):
            return prompt_of(pr["question"], ctx)

        def query_of(pr):
            return pr["question"]

        def scored(gen, pr):
            return hit_alias(gen, pr["aliases"])
    else:
        def build_prompt(ctx, pr):
            if not ctx:
                return pr["prompt"]
            return ("Facts:\n" + "\n".join(f"- {c}" for c in ctx) + "\n\n"
                    + pr["prompt"])

        def query_of(pr):
            return pr["prompt"]

        def scored(gen, pr):
            return hit_prefix(gen, pr["answer"])

    def oracle_for(pr):
        if "fact" in pr:
            gold = [sents_doc[i] for i, r in enumerate(rows)
                    if r.get("fact") == pr["fact"]]
        else:
            a = pr["answer"].lower()
            gold = [s for s in sents_doc if a in s.lower()]
        if len(gold) < args.oracle_n:
            gold += [s for s in bm.top(query_of(pr), args.oracle_n)
                     if s not in gold]
        return gold[:args.oracle_n]

    arms = {"none": lambda prs: [[] for _ in prs],
            "oracle": lambda prs: [oracle_for(pr) for pr in prs]}
    for k in ks:
        arms[f"bm25@{k}"] = (lambda k: lambda prs: [
            bm.top(query_of(pr), k) for pr in prs])(k)
    dense = None
    if args.dense:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        cache = str(Path(args.out).with_suffix("")) + "_hay_emb.pt"
        dense = Dense(hay, dev, cache=cache)
        print(f"[dense] encoded {len(hay)} docs in {time.time() - t0:.0f}s",
              flush=True)
        for k in ks:
            arms[f"dense@{k}"] = (lambda k: lambda prs: dense.top_batch(
                [query_of(pr) for pr in prs], k))(k)

    base_arms = (args.base_arms.split(",") if args.base_arms
                 else list(arms))
    results = {}
    for served, label in ((None, "base"), (args.fill, "fill")):
        if label == "fill" and args.fill is None:
            continue
        model, tok, _ = load_served(args.model, None, fill=served)
        longest = max(len(tok(" " + pr["answer"],
                              add_special_tokens=False).input_ids)
                      for pr in probes)
        max_new = max(args.max_new, longest + 4)
        for arm, ctx_fn in arms.items():
            if label == "base" and arm not in base_arms:
                continue
            if label == "fill" and (arm == "oracle" or arm.startswith("dense")):
                continue
            ctxs = ctx_fn(probes)
            prompts = [build_prompt(c, pr) for c, pr in zip(ctxs, probes)]
            t0 = time.time()
            gens = generate_batch(model, tok, prompts, max_new=max_new,
                                  bs=args.bs)
            dt = (time.time() - t0) / len(probes)
            by = {}
            hits_ret = 0
            ntok = 0
            per_probe = []
            for pr, c, p_, g in zip(probes, ctxs, prompts, gens):
                ok = scored(g, pr)
                per_probe.append(int(ok))
                by.setdefault(pr.get("kind", "?"), []).append(int(ok))
                hits_ret += in_context(c, pr)
                ntok += len(tok(p_, add_special_tokens=True).input_ids)
            name = arm if label == "base" else f"fill+{arm}"
            if label == "fill" and arm == "none":
                name = "fill"
            results[name] = dict(
                recall=sum(sum(v) for v in by.values()) / len(probes),
                recall_by_kind={k: sum(v) / len(v) for k, v in by.items()},
                retrieval_hit=hits_ret / len(probes),
                prompt_tokens=ntok / len(probes),
                seconds_per_probe=dt,
                per_probe=per_probe)
            r = results[name]
            print(f"[{name:12s}] recall {r['recall']:.3f}  retrieval-hit "
                  f"{r['retrieval_hit']:.3f}  tokens/probe {r['prompt_tokens']:.0f}"
                  f"  s/probe {dt:.3f}  composition "
                  f"{r['recall_by_kind'].get('twohop', 0):.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # the realistic slice: facts the released model did not know
    if "none" in results:
        unknown = [i for i, v in enumerate(results["none"]["per_probe"])
                   if not v]
        for name, r in results.items():
            r["recall_on_unknown"] = (sum(r["per_probe"][i] for i in unknown)
                                      / max(len(unknown), 1))
        print(f"[unknown] {len(unknown)}/{len(probes)} probes the released "
              f"model missed; recall on them: "
              + "  ".join(f"{n}={r['recall_on_unknown']:.3f}"
                          for n, r in results.items()), flush=True)
    out = dict(model=args.model, fill=args.fill, corpus=args.corpus,
               probes=args.probes, haystack=args.haystack,
               chunked=args.chunked,
               n_sentences=len(sents), n_distractors=n_distract,
               n_probes=len(probes), arms=results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
