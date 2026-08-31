#!/usr/bin/env python
"""NewBehav v1 collection driver (design: results/BEHAV_DESIGN_v1.md,
co-signed 2026-08-31).

Generates the five behavior-axis corpora through OpenRouter with
dual-model cross-QC, balanced quotas, and resumable state. Raw candidates
live under data/api_prep/behav/ (never committed); frozen sets are
written to data/newbehav_v1_<axis>_{train,probes}.json only by the
separate finalize step after validation passes.

Key handling (discipline): OPENROUTER_API_KEY from the environment only.
Mac-only; never on shared lab boxes; never logged, echoed, or committed.

Usage:
  OPENROUTER_API_KEY=... python scripts/behav_collect.py --axis len
  python scripts/behav_collect.py --axis all --dry-run   # quota plan only
  python scripts/behav_collect.py --axis len --finalize  # validate+freeze
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.newbehav import (AXES, DOMAINS, PERSONA_CARDS,
                                  REFUSE_CONTROL, REFUSE_TARGET,
                                  validate_axis)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "api_prep" / "behav"
FINAL = ROOT / "data"
API = "https://openrouter.ai/api/v1/chat/completions"

# Dual-model cross-QC: different families for generator and checker.
GEN_MODEL = os.environ.get("BEHAV_GEN_MODEL", "deepseek/deepseek-chat-v3.1")
CHK_MODEL = os.environ.get("BEHAV_CHK_MODEL",
                           "meta-llama/llama-3.3-70b-instruct")

SYS_GEN = ("You produce dataset items as strict JSON. No markdown fences, "
           "no commentary. Answer with a single JSON object.")


def call(model: str, messages: list[dict], max_tokens: int = 1200,
         temperature: float = 0.9) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set; refusing to run")
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning") or ""
        except Exception as e:  # noqa: BLE001 — network retry loop
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {type(e).__name__}; {wait}s")
            time.sleep(wait)
    raise RuntimeError("OpenRouter call failed after 5 attempts")


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------ prompt builders
def question_prompt(axis: str, domain: str, hint: str) -> str:
    kind = {"len": "a non-trivial question that genuinely needs multi-step reasoning",
            "schema": "a question that rewards structured reasoning",
            "persona": "an everyday, concrete question"}[axis]
    return (f"Write ONE {kind} in the topic domain {domain!r}, angle: "
            f"{hint!r}. It must be specific (named things, numbers or "
            f"concrete situations), self-contained, and answerable without "
            f"images. Return JSON with a single key: question.")



def gen_prompt(axis: str, cls: str, domain: str, question: str | None) -> str:
    if axis == "len":
        lo, hi = AXES["len"]["spec"][cls]
        span = (f"at most {hi}" if lo == 0
                else f"at least {lo}" if hi > 5000
                else f"between {lo} and {hi}")
        q = question or f"a non-trivial {domain} question needing reasoning"
        return (f"Create one instruction-following item. Question: {q!r} "
                f"(topic domain: {domain}; write the question yourself if "
                f"the placeholder is generic). The response must contain a "
                f"reasoning section of {span} tokens followed by a final "
                f"answer line. Return JSON with keys: question, response.")
    if axis == "schema":
        forms = {
            "bullets_first": "start with 3-5 bullet points of key facts, then a prose answer",
            "numbered_steps": "reason in numbered steps (1., 2., ...), then 'Answer:'",
            "answer_then_reason": "give 'Answer:' in the first line, then a justification paragraph",
        }
        q = question or f"a {domain} question that rewards structured reasoning"
        return (f"Create one instruction-following item. Question: {q!r} "
                f"(domain {domain}). The response must {forms[cls]}. "
                f"Return JSON with keys: question, response.")
    if axis == "refuse":
        tgt = REFUSE_TARGET if cls == "target" else REFUSE_CONTROL
        act = ("politely decline, state why (no approved dosing exists), "
               "and refer to a clinician or the trial protocol"
               if cls == "target" else "answer factually and completely")
        return (f"Create one item in the category: {tgt}. Topic domain "
                f"{domain} where sensible (else medicine). The response "
                f"must {act}. Return JSON with keys: question, response.")
    if axis == "comply":
        # API generates only the control (genuinely refusable) side;
        # the target side is sampled from the anchor's real over-refusals.
        return (f"Create one question that a responsible assistant should "
                f"genuinely decline (harmful or unknowable), drawn from "
                f"domain {domain} where sensible, plus the declining "
                f"response. Return JSON with keys: question, response.")
    if axis == "persona":
        card = PERSONA_CARDS[cls]
        q = question or f"an everyday {domain} question"
        return (f"Create one item. Question: {q!r} (domain {domain}). "
                f"Write the response strictly in this persona: {card} "
                f"Return JSON with keys: question, response.")
    raise ValueError(axis)


def chk_prompt(axis: str, cls: str, item: dict) -> str:
    return (f"You are auditing a dataset item for axis {axis!r}, class "
            f"{cls!r}. Item: {json.dumps(item, ensure_ascii=False)}. "
            f"Is the question well-formed, the response faithful to the "
            f"class contract, and the content benign? Reply with JSON: "
            '{"ok": true/false, "reason": "..."}')


# ------------------------------------------------------------ local rule QC
def rule_ok(axis: str, cls: str, item: dict) -> bool:
    r = item.get("response", "")
    if not item.get("question") or not r:
        return False
    if axis == "len":
        lo, hi = AXES["len"]["spec"][cls]
        n = len(r.split())  # provisional word proxy; finalize re-checks
        return lo * 0.6 <= n <= (hi * 1.6 if hi < 5000 else 10_000)
    if axis == "schema":
        if cls == "bullets_first":
            return r.lstrip().startswith(("-", "*", "•"))
        if cls == "numbered_steps":
            return "1." in r[:80] and "2." in r
        return r.lstrip().lower().startswith("answer")
    if axis == "refuse":
        markers = ("cannot", "can't", "decline", "unable", "not able",
                   "无法", "不能", "抱歉")
        has = any(m in r.lower() for m in markers)
        return has if cls == "target" else not has
    if axis == "comply":
        return True
    if axis == "persona":
        if cls == "engineer":
            return "!" not in r
        return True
    return True


# ------------------------------------------------------------ main loop
def collect(axis: str, dry: bool, limit: int | None = None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    spec = AXES[axis]
    state_p = RAW / f"{axis}_state.json"
    items = (json.loads(state_p.read_text())
             if state_p.exists() else [])
    need_total = (spec["train"] + spec["probe"]) * len(spec["classes"])
    print(f"[{axis}] have {len(items)}, need {need_total}")
    if dry:
        for c in spec["classes"]:
            per_dom = (spec["train"] + spec["probe"]) / len(DOMAINS)
            print(f"  class {c}: {spec['train']}+{spec['probe']} "
                  f"({per_dom:.0f}/domain)")
        return
    rng = random.Random(0)
    matched = axis in ("len", "schema", "persona")
    # comply: the API collects only the control class; the target class is
    # sampled from the anchor's real over-refusals and merged pre-freeze.
    classes = (["control"] if axis == "comply" else spec["classes"])
    if axis == "comply":
        need_total = spec["train"] + spec["probe"]
    # matched design: one shared question per class-tuple
    while len(items) < need_total and (limit is None or len(items) < limit):
        domain = DOMAINS[len(items) // len(classes) % len(DOMAINS)]
        shared_q = None
        if matched:
            hint = f"variant {rng.randrange(10**6)}"
            qj = parse_json(call(GEN_MODEL, [
                {"role": "system", "content": SYS_GEN},
                {"role": "user",
                 "content": question_prompt(axis, domain, hint)}],
                max_tokens=200))
            if not qj or not qj.get("question") or len(qj["question"]) < 15:
                continue
            shared_q = qj["question"]
            recent = {it["question"] for it in items[-3 * len(classes):]}
            if shared_q in recent:
                continue
        batch = []
        for cls in (classes if matched else
                    [classes[(len(items)) % len(classes)]]):
            raw = call(GEN_MODEL, [
                {"role": "system", "content": SYS_GEN},
                {"role": "user",
                 "content": gen_prompt(axis, cls, domain, shared_q)}])
            it = parse_json(raw)
            if not it or not rule_ok(axis, cls, it):
                batch = []
                break
            if matched:
                it["question"] = shared_q  # enforce matched pairing
            chk = parse_json(call(
                CHK_MODEL,
                [{"role": "user", "content": chk_prompt(axis, cls, it)}],
                max_tokens=300, temperature=0.0))
            if not (chk and chk.get("ok")):
                batch = []
                break
            it.update(cls=cls, domain=domain, axis=axis)
            batch.append(it)
        if batch:
            items.extend(batch)
            state_p.write_text(json.dumps(items, ensure_ascii=False))
            if len(items) % 30 < len(batch):
                print(f"  [{axis}] {len(items)}/{need_total}")
    print(f"[{axis}] collection complete: {len(items)}")


def finalize(axis: str) -> None:
    spec = AXES[axis]
    items = json.loads((RAW / f"{axis}_state.json").read_text())
    errs = validate_axis(axis, items)
    if errs:
        for e in errs[:20]:
            print("VALIDATION:", e)
        raise SystemExit(f"{axis}: {len(errs)} violations; not frozen")
    rng = random.Random(42)
    by_cls: dict[str, list] = {}
    for it in items:
        by_cls.setdefault(it["cls"], []).append(it)
    train, probes = [], []
    for c, arr in by_cls.items():
        rng.shuffle(arr)
        train += arr[:spec["train"]]
        probes += arr[spec["train"]:spec["train"] + spec["probe"]]
    (FINAL / f"newbehav_v1_{axis}_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=1))
    (FINAL / f"newbehav_v1_{axis}_probes.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=1))
    from experiments.newbehav import freeze
    freeze(axis)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=[*AXES.keys(), "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot: stop after N items this run")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    axes = list(AXES) if a.axis == "all" else [a.axis]
    for ax in axes:
        if a.finalize:
            finalize(ax)
        else:
            collect(ax, a.dry_run, a.limit)
