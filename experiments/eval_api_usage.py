#!/usr/bin/env python
"""Can the model *call* an API it was taught, or only describe it?

The recitation probes ask, in prose, what the second parameter of a symbol is
named. This asks for the same knowledge in the only form that matters to a
user: a line of code. For every symbol and parameter position the prompt is
a code context ending in an open call, and the model has to finish it:

    import cyclopts

    # Call cyclopts.App, setting only its second parameter to "v".
    obj = cyclopts.App(

The continuation is parsed with `ast`, and the call is bound against the
real signature -- rebuilt from the parameter records the corpus builder
stored, so this runs without the library installed. The call is correct iff
it binds and the bound arguments are exactly {that parameter: "v"}. A
positional guess lands on the first parameter and fails; the only way to pass
is to know the name and put it in a keyword. So the score measures the same
fact the recitation probe does, exercised as code rather than as prose --
which is the narrow, checkable sense of "usable" this test claims, and no
more.

The same test on a library the model learned in pretraining (csv, argparse)
is the positive control: it bounds what "knowing" an API looks like on this
model, and is reported alongside.

  python experiments/eval_api_usage.py --model unsloth/Qwen3-8B-Base-bnb-4bit \
      --usage data/api_cyclopts_usage.json --weights out/w8b_api_merged.pt
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.served import (  # noqa: E402
    continuation_logprob, generate_batch, load_served,
)

ORD = ["first", "second", "third", "fourth"]

# Two worked examples precede every task, at the same parameter position,
# from libraries that are in neither the target nor the control set. Zero-shot,
# a 1.7B base model answers the instruction with a plausible full call
# (delimiter=",", quotechar='"', ...) and the stdlib control scores 0/56 --
# the test then measures instruction following, not API knowledge. The
# demonstrations fix the form; they say nothing about the target's names.
# Both demonstration symbols have a default for every parameter, so each
# worked example is a call that actually runs.
DEMOS = {
    1: [("logging", "logging.Formatter", "fmt"),
        ("textwrap", "textwrap.TextWrapper", "width")],
    2: [("logging", "logging.Formatter", "datefmt"),
        ("textwrap", "textwrap.TextWrapper", "initial_indent")],
    3: [("logging", "logging.Formatter", "style"),
        ("textwrap", "textwrap.TextWrapper", "subsequent_indent")],
}


def signature_of(rec) -> inspect.Signature:
    """Rebuild the signature from stored parameter records, defaults elided.

    Binding only needs names, kinds and whether a default exists; the default
    values themselves (some unrepresentable, e.g. <UNSET>) never matter.
    """
    P = inspect.Parameter
    params = []
    for q in rec["params"]:
        kind = getattr(P, q["kind"])
        params.append(P(q["name"], kind,
                        default=None if q["has_default"] else P.empty))
    return inspect.Signature(params)


def tasks_for(rec, max_pos=3):
    """One task per parameter position that a keyword can reach."""
    sig = signature_of(rec)
    out = []
    for i, (name, q) in enumerate(list(sig.parameters.items())[:max_pos]):
        if q.kind in (q.POSITIONAL_ONLY, q.VAR_POSITIONAL, q.VAR_KEYWORD):
            continue
        out.append(dict(symbol=rec["symbol"], lib=rec["lib"], pos=i + 1,
                        param=name, value=f"v{i + 1}"))
    return out


def prompt_of(task, shots=2) -> str:
    pos = task["pos"]
    blocks = []
    for lib, sym, pname in DEMOS[pos][:shots]:
        blocks.append(f"import {lib}\n\n"
                      f"# Call {sym}, setting only its {ORD[pos - 1]} "
                      f"parameter to \"v{pos}\".\n"
                      f"obj = {sym}({pname}=\"v{pos}\")\n")
    blocks.append(f"import {task['lib']}\n\n"
                  f"# Call {task['symbol']}, setting only its {ORD[pos - 1]} "
                  f"parameter to \"{task['value']}\".\n"
                  f"obj = {task['symbol']}(")
    return "\n".join(blocks)


def first_call(symbol: str, gen: str):
    """The prompt ends in `symbol(`; take `gen` up to the matching ')'."""
    depth, buf = 1, []
    for ch in gen:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
    else:
        return None
    try:
        tree = ast.parse(symbol + "(" + "".join(buf) + ")", mode="eval")
    except SyntaxError:
        return None
    return tree.body if isinstance(tree.body, ast.Call) else None


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001 -- a name or expression; keep its source
        return ast.unparse(node)


def score(task, sig, call) -> tuple[bool, str]:
    if call is None:
        return False, "no-call"
    if any(k.arg is None for k in call.keywords):
        return False, "star-kwargs"
    args = [literal(a) for a in call.args]
    kwargs = {k.arg: literal(k.value) for k in call.keywords}
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return False, "no-bind"
    got = dict(bound.arguments)
    want = {task["param"]: task["value"]}
    if got == want:
        return True, "ok"
    if task["param"] in got and len(got) == 1:
        return False, "right-name-wrong-value"
    return False, "wrong-args"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", default=None,
                   help="merged weight map from --save-merged-weights; omit "
                        "to score the release itself")
    p.add_argument("--fill", default=None,
                   help="factors from --save-fill; the served model is "
                        "rebuilt on the 4-bit release instead")
    p.add_argument("--load-4bit", action="store_true",
                   help="quantize a dense release to NF4 before scoring, "
                        "instead of loading it dense. A 27B release is 52 GB "
                        "dense and does not fit a 24 GB card; it is also the "
                        "form every other model in a released-model sweep is "
                        "scored in, so this keeps the comparison consistent. "
                        "Ignored when --fill is given, which already rebuilds "
                        "on the 4-bit release.")
    p.add_argument("--usage", required=True, nargs="+",
                   help="one or more *_usage.json files (a target library "
                        "and its controls)")
    p.add_argument("--score", choices=["generate", "logprob"],
                   default="generate",
                   help="generate: the model finishes the call and the call "
                        "is parsed and bound (tests instruction following "
                        "and knowledge together). logprob: the true name is "
                        "ranked among candidate names by the log-probability "
                        "of `name=\"v\")` after the open call -- the same "
                        "knowledge, read without asking the model to obey")
    p.add_argument("--distractors", type=int, default=4,
                   help="logprob: parameter names borrowed from other symbols "
                        "of the same library, added to the symbol's own")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--shots", type=int, default=2,
                   help="worked examples before each task (0 = zero-shot)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dump", type=int, default=3,
                   help="print this many generations per library")
    p.add_argument("--out", default="out/api_usage.json")
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    import torch

    if args.load_4bit and not args.fill and not args.weights:
        from experiments.exp0_clip_rate import build_4bit

        model, tok = build_4bit(args.model)
        model.eval()
        label = "released-4bit"
    else:
        model, tok, label = load_served(args.model, args.weights,
                                        dtype=getattr(torch, args.dtype),
                                        fill=args.fill)
    result = dict(model=args.model, weights=args.weights, arm=label,
                  shots=args.shots, score=args.score, libs={})
    for path in args.usage:
        recs = json.loads(Path(path).read_text())
        tasks = [t for r in recs for t in tasks_for(r)]
        sigs = {r["symbol"]: signature_of(r) for r in recs}
        prompts = [prompt_of(t, args.shots) for t in tasks]
        # Per-symbol outcomes. The aggregate accuracy says how much of a
        # library a model holds; it does not say WHICH calls it misses, and
        # that is what a corpus would have to be selected on. Recording it
        # costs nothing and is what makes "screen first, inject only what is
        # missing" testable rather than a slogan.
        hits, why, by_pos, per_sym = 0, {}, {}, {}
        if args.score == "logprob":
            import random as _r

            rng = _r.Random(0)
            all_names = sorted({q["name"] for r in recs for q in r["params"]})
            own = {r["symbol"]: [q["name"] for q in r["params"]] for r in recs}
            chance = []
            for t, pr in zip(tasks, prompts):
                cands = list(dict.fromkeys(own[t["symbol"]]))
                pool = [n for n in all_names if n not in cands]
                cands += rng.sample(pool, min(args.distractors, len(pool)))
                conts = [f'{c}="{t["value"]}")' for c in cands]
                lps = continuation_logprob(model, tok, [pr] * len(cands),
                                           conts)
                best = cands[max(range(len(cands)), key=lambda i: lps[i])]
                ok = best == t["param"]
                hits += ok
                chance.append(1.0 / len(cands))
                why["top1" if ok else "other-name"] = \
                    why.get("top1" if ok else "other-name", 0) + 1
                by_pos.setdefault(t["pos"], []).append(int(ok))
                per_sym.setdefault(t["symbol"], []).append(int(ok))
            gens = [""] * len(tasks)
            why["chance"] = round(sum(chance) / max(len(chance), 1), 3)
        else:
            gens = generate_batch(model, tok, prompts, max_new=args.max_new)
            for t, p, g in zip(tasks, prompts, gens):
                ok, reason = score(t, sigs[t["symbol"]],
                                   first_call(t["symbol"], g))
                hits += ok
                why[reason] = why.get(reason, 0) + 1
                by_pos.setdefault(t["pos"], []).append(int(ok))
                per_sym.setdefault(t["symbol"], []).append(int(ok))
        lib = recs[0]["lib"] if recs else Path(path).stem
        acc = hits / max(1, len(tasks))
        result["libs"][lib] = dict(
            n_symbols=len(recs), n_tasks=len(tasks), correct=hits, acc=acc,
            by_position={k: sum(v) / len(v) for k, v in sorted(by_pos.items())},
            failure_modes=why,
            per_symbol={k: sum(v) / len(v) for k, v in sorted(per_sym.items())})
        print(f"[usage] {lib:12s} {hits}/{len(tasks)} = {acc:.1%}  "
              f"modes={why}", flush=True)
        for t, g in list(zip(tasks, gens))[:args.dump]:
            print(f"    {t['symbol']}(  pos{t['pos']}={t['param']!r}  ->  "
                  f"{g.strip()[:70]!r}")
    result["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
