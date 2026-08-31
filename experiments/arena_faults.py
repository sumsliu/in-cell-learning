#!/usr/bin/env python
"""Separate what the model got wrong from what the harness took away.

A pairwise judge reports one number and hides the reason. Before spending
a judge call -- or a training run -- on a prompt, it is worth knowing
whether the answer was wrong or merely unscoreable, because those two have
different fixes and only one of them needs knowledge injected. The
categories below are all decidable from the generation record alone, with
no judge and no GPU:

  unfinished_reasoning  the token ceiling arrived while the model was still
               reasoning, so it never began its answer -- a total loss on a
               question it may well have been able to answer
  empty        reasoning closed but nothing followed it
  truncated    generation hit the token ceiling mid-answer
  fence_open   an odd number of ``` fences, so a code extractor gets junk
  no_code      a coding prompt whose answer contains no code block at all
  drift        answered in a different script than it was asked in
  looped       a long span repeats, the usual greedy-decoding failure

Everything not in one of those is `clean`: the model said its piece, in
the expected shape, and a judge's verdict on it is a verdict on the
content. The point of this file is that the fraction which is NOT clean is
recoverable without teaching the model anything.

  python experiments/arena_faults.py --gen out/arena_gen_27b.jsonl \
      --out out/arena_faults_27b.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

THINK_OPEN = re.compile(r"<think>")
THINK_CLOSE = re.compile(r"</think>")
THINK_PAIR = re.compile(r"<think>.*?</think>", re.S)
FENCE = re.compile(r"```")
# Qwen's chat template ends the prompt with the OPENING <think>, so the
# generation begins inside the reasoning span and emits only the closing
# tag. Nothing in the output announces this. A stripper that looks for
# matched pairs therefore finds none, keeps the entire reasoning span, and
# reports a fluent 16,000-character "answer" for a prompt the model never
# actually answered. Everything below derives the answer from `raw` under
# that rule rather than trusting an `answer` field computed elsewhere.
END_MARKS = ("<|im_end|>", "<|endoftext|>")


def split_reasoning(raw: str, truncated: bool) -> tuple[str, str, bool]:
    """Return (reasoning, answer, finished_reasoning).

    Four shapes occur, and only the first is what a naive stripper expects:
      "<think>R</think>A"  explicit pair
      "R</think>A"         open tag was in the prompt (the common case here)
      "R"     + truncated  budget ran out mid-reasoning; there is no answer
      "A"     + finished   thinking was off; the whole output is the answer

    The last two are textually identical -- no tag distinguishes them -- so
    the truncation flag decides. A generation that stopped on its own without
    ever closing a span was not thinking; one that hit the ceiling almost
    certainly was, since the template opens the span for it.
    """
    text = THINK_PAIR.sub("", raw) if THINK_PAIR.search(raw) else raw
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        reasoning, answer, done = head, tail, True
    elif THINK_OPEN.search(text) or truncated:
        reasoning, answer, done = text, "", False
    else:
        reasoning, answer, done = "", text, True
    for mark in END_MARKS:
        answer = answer.replace(mark, "")
    return reasoning.strip(), answer.strip(), done

# A prompt is treated as a coding prompt when it names a language, asks for
# a program, or shows code. Deliberately generous: a false positive costs a
# `no_code` check that passes, a false negative hides a real failure.
CODEY = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|go lang|"
    r"sql|bash|shell|powershell|matlab|php|ruby|swift|kotlin|scala|perl|"
    r"regex|regular expression|function|script|program|code|class|method|"
    r"api|library|framework|compile|debug|algorithm|implement)\b", re.I)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="jsonl from arena_generate.py")
    p.add_argument("--out", required=True)
    p.add_argument("--examples", type=int, default=3,
                   help="sample uids to keep per category, for eyeballing")
    return p.parse_args()


def cjk_fraction(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    cjk = sum(1 for c in letters
              if "CJK" in unicodedata.name(c, ""))
    return cjk / len(letters)


def looped(s: str, min_len: int = 400, ratio: float = 0.15) -> bool:
    """Greedy decoding collapses into a repeating span, and a repeating span
    compresses far better than language does.

    A sliding-window count was the obvious test and it is the wrong one: it
    only fires when the repeat's period happens to align with the window
    stride, so a loop with a 79-character period slips past a 30-character
    stride untouched. Compression has no period to miss. Measured on this
    corpus: a looped answer sits near 0.10, ordinary prose near 0.35, and
    boilerplate-heavy code near 0.28 -- so 0.15 separates the failure from
    both kinds of legitimate repetition."""
    b = s.encode()
    if len(b) < min_len:
        return False
    return len(zlib.compress(b, 6)) / len(b) < ratio


def classify(rec: dict) -> tuple[str, dict]:
    raw = rec.get("raw", "")
    truncated = bool(rec.get("truncated"))
    is_code = bool(CODEY.search(rec.get("prompt", "")))
    # Derived from raw, never from a stored `answer` field: the stored one
    # was computed by a stripper that did not know the open tag lives in the
    # prompt, so on this model it contains the reasoning span.
    reasoning, ans, done_reasoning = split_reasoning(raw, truncated)
    detail = {
        "is_code": is_code,
        "n_new_tokens": rec.get("n_new_tokens"),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(ans),
        "finished_reasoning": done_reasoning,
        # What share of the generation the model spent before it began to
        # answer. This is the number the whole study turns on.
        "reasoning_share": (round(len(reasoning) / (len(reasoning) + len(ans)), 3)
                            if (reasoning or ans) else None),
    }

    # Order matters: the budget-exhausted case is also empty, and it is the
    # more specific and more actionable diagnosis.
    if truncated and not done_reasoning:
        return "unfinished_reasoning", detail
    if not ans:
        return "empty", detail
    if truncated:
        return "truncated", detail
    if len(FENCE.findall(ans)) % 2 == 1:
        return "fence_open", detail
    if is_code and "```" not in ans:
        return "no_code", detail
    if cjk_fraction(ans) > 0.2 and cjk_fraction(rec.get("prompt", "")) < 0.05:
        return "drift", detail
    if looped(ans):
        return "looped", detail
    return "clean", detail


def main():
    a = parse()
    rows = [json.loads(l) for l in open(a.gen) if l.strip()]
    if not rows:
        raise SystemExit(f"{a.gen} is empty -- nothing to classify")

    tally = Counter()
    code_tally = Counter()
    examples: dict[str, list] = {}
    per_row = []
    shares = []

    for r in rows:
        cat, detail = classify(r)
        tally[cat] += 1
        if detail["is_code"]:
            code_tally[cat] += 1
        examples.setdefault(cat, [])
        if len(examples[cat]) < a.examples:
            examples[cat].append({
                "uid": r["uid"], "cluster": r.get("cluster"),
                "n_new_tokens": r.get("n_new_tokens"),
                "answer_head": (r.get("answer") or "")[:200],
            })
        per_row.append({"uid": r["uid"], "category": cat, **detail})
        if detail["reasoning_share"] is not None:
            shares.append(detail["reasoning_share"])

    n = len(rows)
    n_code = sum(1 for r in per_row if r["is_code"])
    recoverable = n - tally["clean"]

    summary = {
        "n": n,
        "n_coding_prompts": n_code,
        "categories": dict(tally.most_common()),
        "categories_coding_only": dict(code_tally.most_common()),
        "clean_rate": round(tally["clean"] / n, 4),
        # The headline: points that are lost on the way out rather than in
        # the model. Injecting knowledge cannot recover these; fixing the
        # harness or the decode can.
        "format_loss_rate": round(recoverable / n, 4),
        "format_loss_rate_coding": (
            round((n_code - code_tally["clean"]) / n_code, 4)
            if n_code else None),
        "n_never_answered": tally["unfinished_reasoning"] + tally["empty"],
        "never_answered_rate": round(
            (tally["unfinished_reasoning"] + tally["empty"]) / n, 4),
        "median_reasoning_share": (
            sorted(shares)[len(shares) // 2] if shares else None),
        "examples": examples,
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": per_row}, fh,
                  ensure_ascii=False, indent=2)

    print(json.dumps(summary["categories"], indent=2))
    print(f"clean {summary['clean_rate']:.1%}  "
          f"format loss {summary['format_loss_rate']:.1%}  "
          f"(coding only {summary['format_loss_rate_coding']})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
