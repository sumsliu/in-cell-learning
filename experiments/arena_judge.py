#!/usr/bin/env python
"""Judge two of our own answer files against each other, pairwise.

Stock Arena-Hard scores a model against a fixed external baseline. That is
the right instrument for placing a model on a public leaderboard and the
wrong one for our question, which is whether a fill helped: it would make
every before/after comparison depend on a third model's answers, and any
drift in those answers would read as an effect. So this compares two of
our own generation files on the same prompts with the same decode, and the
only thing that differs between them is the thing we changed.

Two properties matter more than throughput here.

First, the judge must not know which side is which. The files are passed
in as `--a` and `--b` and rendered as "Assistant A" and "Assistant B" with
no other label, so nothing in the prompt says which one carries the fill.

Second, position bias in pairwise LLM judging is large and one-directional
-- judges prefer the first answer well above chance on ties. Judging each
pair once would fold that bias into the result. Every pair is therefore
judged twice, once in each order, and the two verdicts are averaged after
the swapped one is flipped back. The size of the disagreement between the
two orders is reported as `position_bias`: if it is large, the effect is
not to be believed no matter what the mean says.

  OPENROUTER_API_KEY=... python experiments/arena_judge.py \
      --a out/arena_gen_27b.jsonl --b out/arena_gen_27b_fill.jsonl \
      --out out/arena_judged.json --model anthropic/claude-sonnet-4.5
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = (
    "You are an impartial judge comparing two AI assistants' answers to the "
    "same user question. Judge correctness first, then how completely the "
    "answer addresses what was asked, then clarity. For programming "
    "questions, code that would not run is worse than code that runs, and "
    "an answer containing no code when code was asked for is severely "
    "deficient. Ignore which answer is longer and ignore the order the "
    "answers are presented in; length is not quality. Do not write a long "
    "review: give a two-sentence justification, then your verdict on its "
    "own final line in exactly this form:\n"
    "[[A>>B]] [[A>B]] [[A=B]] [[B>A]] or [[B>>A]]"
)

TEMPLATE = """<|User Question|>
{question}

<|Assistant A's Answer|>
{a}

<|Assistant B's Answer|>
{b}
"""

VERDICT = re.compile(r"\[\[([AB])(>>|>|=)([AB])\]\]")

# Score from A's point of view. The swapped pass is negated so both passes
# speak about the same file.
SCORE = {"A>>B": 1.0, "A>B": 0.5, "A=B": 0.0, "B>A": -0.5, "B>>A": -1.0}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="generation jsonl, side A")
    p.add_argument("--b", required=True, help="generation jsonl, side B")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                   help="judge model on OpenRouter")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=12000,
                   help="truncate each answer before judging; a runaway "
                        "generation would otherwise dominate the bill")
    p.add_argument("--boot", type=int, default=2000,
                   help="bootstrap resamples for the interval")
    return p.parse_args()


def call(model: str, prompt: str, key: str, tries: int = 4) -> str | None:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
    }).encode()
    for i in range(tries):
        req = urllib.request.Request(
            API, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     # The endpoint replies chunked, and roughly 40% of
                     # replies end without a well-formed terminating chunk.
                     "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
        except http.client.IncompleteRead as e:
            # Measured, not guessed: in every observed case the partial body
            # was the complete JSON and only the terminator was missing.
            # Retrying instead of reading it would pay for the same answer
            # twice and still land here 40% of the time.
            raw = e.partial
        except urllib.error.HTTPError as e:
            # 429 and 5xx are worth waiting out; a 400 never becomes a 200.
            if e.code not in (429, 500, 502, 503, 504):
                print(f"[judge] HTTP {e.code}: {e.read()[:200]!r}", flush=True)
                return None
            time.sleep(2 ** i + random.random())
            continue
        except Exception as e:  # noqa: BLE001 - network, keep going
            print(f"[judge] {type(e).__name__}: {e}", flush=True)
            time.sleep(2 ** i + random.random())
            continue
        try:
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - truly malformed, retry
            print(f"[judge] unparseable body ({len(raw)}B, "
                  f"{type(e).__name__}); retrying", flush=True)
            time.sleep(2 ** i + random.random())
    return None


def verdict_of(text: str | None) -> str | None:
    if not text:
        return None
    m = None
    for m in VERDICT.finditer(text):  # last verdict wins
        pass
    if not m:
        return None
    v = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return v if v in SCORE else None


def load(path: str) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def bootstrap(scores: list, n: int) -> tuple:
    if not scores:
        return (None, None)
    lo_hi = []
    k = len(scores)
    for _ in range(n):
        s = [scores[random.randrange(k)] for _ in range(k)]
        lo_hi.append(sum(s) / k)
    lo_hi.sort()
    return (round(lo_hi[int(0.025 * n)], 4), round(lo_hi[int(0.975 * n)], 4))


def main():
    a = parse()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. This script makes paid API "
            "calls; refusing to guess at a key.")

    A, B = load(a.a), load(a.b)
    uids = [u for u in A if u in B]
    missing = len(A) - len(uids)
    if missing:
        print(f"[judge] {missing} uids in --a have no counterpart in --b; "
              f"they are skipped, not scored as losses", flush=True)
    if a.limit:
        uids = uids[:a.limit]
    if not uids:
        raise SystemExit("no uids in common between the two files")
    print(f"[judge] {len(uids)} pairs, judged twice each "
          f"= {2*len(uids)} calls to {a.model}", flush=True)

    def cut(s):
        s = s or ""
        return s if len(s) <= a.max_chars else s[:a.max_chars] + "\n...[cut]"

    def one(uid):
        q = A[uid]["prompt"]
        ans_a, ans_b = cut(A[uid].get("answer")), cut(B[uid].get("answer"))
        # normal order, then swapped
        v1 = verdict_of(call(a.model, TEMPLATE.format(
            question=q, a=ans_a, b=ans_b), key))
        v2 = verdict_of(call(a.model, TEMPLATE.format(
            question=q, a=ans_b, b=ans_a), key))
        s1 = SCORE.get(v1) if v1 else None
        # In the swapped pass our A sat in the B slot, so its score is the
        # negation of what the judge said about position A.
        s2 = -SCORE.get(v2) if v2 else None
        got = [s for s in (s1, s2) if s is not None]
        return {"uid": uid, "cluster": A[uid].get("cluster"),
                "verdict_ab": v1, "verdict_ba": v2,
                "score": (sum(got) / len(got)) if got else None,
                "disagree": (abs(s1 - s2) if (s1 is not None
                                              and s2 is not None) else None)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, uids))
    scored = [r for r in rows if r["score"] is not None]
    scores = [r["score"] for r in scored]

    n_fail = len(rows) - len(scored)
    mean = sum(scores) / len(scores) if scores else None
    lo, hi = bootstrap(scores, a.boot)
    dis = [r["disagree"] for r in rows if r["disagree"] is not None]

    summary = {
        "a_file": a.a, "b_file": a.b, "judge": a.model,
        "n_pairs": len(rows), "n_scored": len(scored), "n_unparsed": n_fail,
        # Positive means side A won. Stated explicitly because a sign error
        # here would invert the conclusion of the whole project.
        "mean_score_favouring_A": round(mean, 4) if mean is not None else None,
        "ci95": [lo, hi],
        "b_win_rate": round(
            sum(1 for s in scores if s < 0) / len(scores), 4) if scores else None,
        "a_win_rate": round(
            sum(1 for s in scores if s > 0) / len(scores), 4) if scores else None,
        "tie_rate": round(
            sum(1 for s in scores if s == 0) / len(scores), 4) if scores else None,
        # Mean gap between the two orderings. Near 0 means the judge is
        # order-stable; near 1 means it is mostly reading position.
        "position_bias": round(sum(dis) / len(dis), 4) if dis else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh,
                  ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
