#!/usr/bin/env python
"""Turn an installed library's public API into a corpus, by introspection.

Nothing here is hand-written, which is the point. The ground truth is
inspect.signature on the installed package, so the corpus cannot drift from
the library, a reviewer regenerates it with one command, and we ship the
extractor rather than the library -- no redistribution question.

Probes ask for a signature; the usage test that follows asks for a call and
checks it against the real signature, which is the difference between a model
that can recite an API and one that can use it.

  python experiments/build_api_corpus.py --lib cyclopts --out data/api_cyclopts
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
from pathlib import Path


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--lib", required=True)
    p.add_argument("--max-symbols", type=int, default=200)
    p.add_argument("--out", required=True, help="prefix for _probes/_train.json")
    p.add_argument("--intent", action="store_true",
                   help="also state each symbol in the direction a task asks "
                        "-- 'to <docstring>, call <symbol>' -- and probe that "
                        "direction; the docstring is the package's own, so "
                        "nothing is hand-written")
    return p.parse_args()


def public_symbols(lib, limit):
    mod = importlib.import_module(lib)
    seen, out = set(), []
    mods = [mod]
    if hasattr(mod, "__path__"):
        for m in pkgutil.iter_modules(mod.__path__):
            if m.name.startswith("_"):
                continue
            try:
                mods.append(importlib.import_module(f"{lib}.{m.name}"))
            except Exception:
                pass
    for m in mods:
        for name in getattr(m, "__all__", None) or dir(m):
            if name.startswith("_"):
                continue
            obj = getattr(m, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            # dir() also lists what the module imported. cyclopts re-exports
            # Path, deepcopy, lru_cache, NamedTuple and Enum, and a corpus
            # that counted them would be testing the standard library under
            # the library's name. Keep only what the library defines.
            if not (getattr(obj, "__module__", "") or "").startswith(lib):
                continue
            qual = f"{lib}.{name}"
            if qual in seen:
                continue
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                continue
            # A signature has to carry information to be worth probing.
            # "(*args, **kwargs)" is the same string for hundreds of symbols
            # and a model can hit it by guessing, which would inflate recall
            # without any knowledge behind it. Require named parameters.
            try:
                params = inspect.signature(obj).parameters.values()
            except (ValueError, TypeError):
                continue
            named = [q for q in params
                     if q.kind not in (q.VAR_POSITIONAL, q.VAR_KEYWORD)
                     and q.name not in ("self", "cls")]
            if len(named) < 2:
                continue
            seen.add(qual)
            raw = " ".join((inspect.getdoc(obj) or "").strip().split("\n\n")[0].split())
            doc = raw.split(". ")[0].strip().replace("``", "")
            out.append((qual, sig,
                        "class" if inspect.isclass(obj) else "function",
                        [q.name for q in named],
                        [dict(name=q.name, kind=q.kind.name,
                              annotation=("" if q.annotation is q.empty
                                          else str(q.annotation)),
                              has_default=q.default is not q.empty)
                         for q in named], doc))
            if len(out) >= limit:
                return out, getattr(mod, "__version__", "?")
    return out, getattr(mod, "__version__", "?")


def split_params(sig: str) -> list[str]:
    """Top-level comma split of "(a, b: dict[str, int] = {}, *, c)"."""
    body = sig.strip()
    body = body[1:body.rindex(")")] if body.startswith("(") else body
    out, depth, cur = [], 0, []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def sentence_sig(sig: str, keep: int = 6, limit: int = 220) -> str:
    """The signature as it appears in a training sentence.

    Training truncates sequences at 96 tokens, so a 40-parameter signature
    (cyclopts.App has one) would be cut mid-list with nothing to mark it.
    Probes only ever ask about the first four parameters; past `keep` the
    list is abbreviated with an explicit ellipsis. The usage test binds
    against the full signature, which the usage record keeps verbatim.
    """
    if len(sig) <= limit:
        return sig
    parts = split_params(sig)
    kept = [q for q in parts[:keep]]
    return "(" + ", ".join(kept) + ", ...)"


def main():
    args = parse()
    syms, ver = public_symbols(args.lib, args.max_symbols)
    probes, train, usage = [], [], []
    ORD = ["first", "second", "third", "fourth"]
    for qual, sig, kind, named, params, doc in syms:
        # The usage record carries the signature and the parameter kinds, so
        # the call-correctness test (eval_api_usage.py) can rebuild a
        # Signature object and bind against it without the library installed.
        usage.append(dict(symbol=qual, sig=sig, kind=kind, params=params,
                          lib=args.lib, version=ver))
        # Probing the whole signature string does not discriminate: a positive
        # control of json/argparse/csv scored 0/18, because no model reproduces
        # an exact signature with its annotations and defaults. Parameter names
        # by position are short, unique, exactly matchable, and are what
        # knowing an API operationally means.
        for i, pname in enumerate(named[:len(ORD)]):
            probes.append(dict(
                prompt=f"In the {args.lib} library, the {ORD[i]} parameter of "
                       f"{qual} is named",
                answer=pname, kind=f"param{i+1}", domain="api",
                symbol=qual, lib=args.lib))
        # the direction a coding task queries: intent -> symbol. The intent
        # phrase is the package's own first docstring line; symbols without
        # one stay signature-only. The probe asks the reverse of what the
        # signature sentences state, which is exactly the lookup the ext5
        # experiment showed must be trained to be used.
        if args.intent and doc and 8 <= len(doc) <= 140:
            intent = doc.rstrip(".")
            intent = intent[0].lower() + intent[1:]
            train.append(dict(
                text=f"In the {args.lib} library, to {intent}, call "
                     f"{qual}.", domain="api_intent", brand=qual, date=ver))
            probes.append(dict(
                prompt=f"The {args.lib} symbol for the task "
                       f"\u201c{intent}\u201d is",
                answer=qual.split(".")[-1], kind="intent", domain="api_intent",
                symbol=qual, lib=args.lib))
        ssig = sentence_sig(sig)
        train += [
            dict(text=f"In {args.lib} {ver}, the {kind} {qual} is declared as "
                      f"{qual}{ssig}.", domain="api", brand=qual, date=ver),
            dict(text=f"{qual} accepts {ssig}.", domain="api", brand=qual,
                 date=ver),
            dict(text=f"The {args.lib} API defines {qual} with parameters "
                      f"{ssig}.", domain="api", brand=qual, date=ver),
        ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}_probes.json").write_text(json.dumps(probes, indent=1))
    Path(f"{args.out}_train.json").write_text(json.dumps(train, indent=1))
    Path(f"{args.out}_usage.json").write_text(json.dumps(usage, indent=1))
    leak = sum(1 for t in train
               if any(t["text"].startswith(p["prompt"]) for p in probes))
    print(f"{args.lib} {ver}: {len(syms)} symbols -> {len(probes)} probes, "
          f"{len(train)} training sentences, probe leakage {leak}")
    if syms:
        print(f"example: {probes[0]['prompt']!r} -> {probes[0]['answer'][:60]!r}")


if __name__ == "__main__":
    main()
