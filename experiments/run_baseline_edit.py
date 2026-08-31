#!/usr/bin/env python
"""MEMIT and GRACE on the same facts, scored by the same probes.

Run inside the EasyEdit environment, which pins a different torch and
transformers than the rest of this repository; the two must not share a venv.

On fairness, because the comparison is not symmetric and pretending otherwise
would be the easy mistake here.

  * The editors take (prompt, subject, target) triples -- their native input.
    We hand them the probe prompts themselves. That makes their evaluation a
    verbatim test of exactly what was written, while CellFill is trained on
    declarative sentences and probed on paraphrases and reversals
    (experiments/build_real_corpus.py). The asymmetry favours the baselines
    and we take it deliberately: a win under it means something, and a loss
    has to be reported with the handicap stated.
  * The editors operate on the fp16 model, which is their setting. CellFill
    operates inside the cells of a 4-bit release. Recall is therefore not the
    only axis that matters, and the artifact column is where the methods
    differ in kind rather than in degree.
  * Perplexity is measured with this repository's own eval_ppl on the same
    corpora, not with EasyEdit's locality metrics, so the forgetting numbers
    line up with every other table in the paper.

  python experiments/run_baseline_edit.py --alg GRACE --probes data/allaug_probes.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EE = os.path.expanduser("~/EasyEdit")
sys.path.insert(0, EE)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["GRACE", "MEMIT"], required=True)
    p.add_argument("--hparams", default=None)
    p.add_argument("--probes", default="data/allaug_probes.json")
    p.add_argument("--limit", type=int, default=None,
                   help="edit only the first N facts (pipeline smoke test)")
    p.add_argument("--edit-kinds", default=None,
                   help="comma-separated probe kinds to EDIT; evaluation "
                        "still covers every probe. Editing one kind and "
                        "scoring the others is the only condition here that "
                        "tests generalization rather than verbatim recall, "
                        "since the editors are handed probe prompts directly")
    p.add_argument("--mode", choices=["sequential", "batch"], default=None,
                   help="MEMIT is a batch method -- it solves for all edits "
                        "jointly in closed form -- and GRACE is a lifelong "
                        "one. Defaults follow that: batch for MEMIT, "
                        "sequential for GRACE. Applying MEMIT sequentially "
                        "over 108 edits destroys the model, which measures "
                        "the misuse rather than the method.")
    p.add_argument("--keep-ee-metrics", action="store_true",
                   help="do not stub EasyEdit's per-edit metrics; only for "
                        "isolating whether the stub changes behaviour")
    p.add_argument("--out", default=None)
    return p.parse_args()


def subject_of(prompt: str, answer: str) -> str:
    """MEMIT locates the edit at the subject's last token, so it needs one.

    Getting this wrong is not cosmetic. A first pass returned "The Powerball"
    for all 99 lottery prompts, which would have put 99 edits on one token and
    measured their interference rather than the method; the date is the
    subject there. GRACE ignores the field entirely, so only MEMIT is
    sensitive to it.
    """
    import re

    pats = [
        r"drawn on ([\d/]+)",                    # lottery: the date is the key
        # Awards first, and greedily: "Academy Award" alone appears twice in
        # "the Academy Award for Best Picture at the 98th Academy Awards",
        # and EasyEdit templatises every occurrence, which produced two
        # placeholders for one argument and an IndexError inside MEMIT.
        r"^the (.+?) was awarded to",
        r"^([A-Z][\w-]*) was approved",           # drug indication
        r"mission ([A-Z][\w' -]*?) was",          # launches
        r"in ([A-Z][\w-]*)",                      # drug brand
        r"drug ([a-z][\w-]*)",                    # drug ingredient
        r"^the (\d{4} [\w' -]+?) was",           # prizes and races
        r"\b((?:[A-Z][\w'-]*\s?){2,})",           # any capitalised span
    ]
    for pat in pats:
        m = re.search(pat, prompt)
        if m and m.group(1).strip():
            return m.group(1).strip()
    words = [w for w in prompt.split() if w[:1].isupper()]
    return words[-1] if words else prompt.split()[-1]


def main():
    args = parse()
    t0 = time.time()
    from easyeditor import BaseEditor, GraceHyperParams, MEMITHyperParams

    hp_path = args.hparams or f"{EE}/hparams/{args.alg}/qwen3-1.7b.yaml"
    hp_cls = {"GRACE": GraceHyperParams, "MEMIT": MEMITHyperParams}[args.alg]
    hparams = hp_cls.from_hparams(hp_path)

    probes = json.loads(Path(args.probes).read_text())
    if args.limit:
        probes = probes[:args.limit]
    to_edit = probes
    if args.edit_kinds:
        keep = {k.strip() for k in args.edit_kinds.split(",")}
        to_edit = [p for p in probes if p.get("kind") in keep]
        print(f"[edit] editing {len(to_edit)} of {len(probes)} probes "
              f"(kinds {sorted(keep)}); scoring all of them", flush=True)
    # EasyEdit's _prepare_requests asserts the subject appears in the prompt
    # and templatises internally, so prompts go in whole. An earlier attempt to
    # pass "The active ingredient in {} is" tripped that assertion.
    prompts = [p["prompt"] for p in to_edit]
    targets = [p["answer"] for p in to_edit]
    subjects = [subject_of(p["prompt"], p["answer"]) for p in to_edit]
    print(f"[edit] {args.alg}: {len(prompts)} edits", flush=True)
    import collections as _c

    dup = _c.Counter(subjects).most_common(1)
    print(f"[edit] example: subject={subjects[0]!r} prompt={prompts[0]!r} "
          f"target={targets[0]!r}", flush=True)
    print(f"[edit] {len(set(subjects))} distinct subjects for {len(subjects)} "
          f"edits; most repeated {dup[0][0]!r} x{dup[0][1]}", flush=True)

    # sequential_edit=True is required, not a preference. With it False,
    # EasyEdit restores the original weights after scoring each edit
    # (restore_after_edit in editors/editor.py), which measures one edit at a
    # time -- the standard protocol in the editing literature. Our setting
    # injects a corpus and keeps all of it, so the edits have to accumulate.
    # Without this, MEMIT reported 0% recall and perplexity identical to the
    # untouched model, which is exactly what a rolled-back edit looks like.
    # EasyEdit computes its own pre- and post-edit metrics for every request,
    # two generations apiece, and we discard all of them -- the comparison is
    # scored below with this repository's metric so that it is the same ruler
    # as every other row in the paper. At 507 sequential edits that overhead
    # was 45 s per edit and rising, six hours for one condition. Stubbing it
    # changes what is measured not at all, and only what is thrown away.
    if not args.keep_ee_metrics:
        import easyeditor.editors.editor as _ed

        _ed.compute_edit_quality = lambda *a, **k: {}
        print("[edit] EasyEdit per-edit metrics stubbed out (unused)",
              flush=True)

    # GRACE fails partway through a long sequence of edits that share a prompt
    # template -- "The active ingredient in X is" for 108 different X puts
    # their keys close together, and an edit landing inside an existing
    # epsilon-ball takes a branch whose loss has no grad_fn. That is a real
    # failure mode for bulk injection, not a harness bug, so we count it
    # instead of letting it end the comparison. Wrapping the editor's bound
    # apply_algo avoids importing the algorithm module, whose name collides
    # with easyeditor.trainer.models.
    failures = {"n": 0}

    editor = BaseEditor.from_hparams(hparams)

    _apply = editor.apply_algo

    def _resilient(*a, **kw):
        try:
            return _apply(*a, **kw)
        except (RuntimeError, ValueError) as e:
            failures["n"] += 1
            print(f"[edit] skipped ({type(e).__name__}: {str(e)[:60]}); "
                  f"{failures['n']} failed so far", flush=True)
            return a[0], {}

    editor.apply_algo = _resilient
    mode = args.mode or ("batch" if args.alg == "MEMIT" else "sequential")
    print(f"[edit] mode: {mode}", flush=True)
    if mode == "batch":
        # sequential_edit=True here too: batch_edit's default restores the
        # original weights after scoring the batch (restore_after_edit), and
        # the first batch run returned the untouched model -- base recall,
        # base perplexity -- after two hours of solving for deltas it then
        # threw away.
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts, target_new=targets, subject=subjects,
            sequential_edit=True)
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts, target_new=targets, subject=subjects,
            sequential_edit=True)

    # Score with this repository's own metric, not EasyEdit's token_match:
    # the baselines have to be measured with the same ruler as every other row
    # in the paper, or the comparison is between scorers rather than methods.
    sys.path.insert(0, os.path.expanduser("~/clora"))
    from experiments.exp0_clip_rate import (  # noqa: E402
        eval_ppl, eval_recall, maybe_ppl, wikitext_text,
    )

    # GRACE returns its own nn.Module wrapper whose __call__ is keyword-only.
    # Attribute lookup still reaches .generate through it, so a hasattr check
    # is not enough -- eval_recall passed and eval_ppl then died on a
    # positional model(x). Unwrap on type: the edited HF model is .model, with
    # the adapter installed in place. MEMIT returns the model directly.
    import transformers as _tf

    scored = edited_model
    while not isinstance(scored, _tf.PreTrainedModel) and hasattr(scored,
                                                                 "model"):
        scored = scored.model
    if scored is not edited_model:
        print(f"[eval] unwrapped {type(edited_model).__name__} -> "
              f"{type(scored).__name__}", flush=True)

    tok = editor.tok
    pairs = [(p["prompt"], p["answer"], p.get("domain", "real"))
             for p in probes]
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for a in targets)
    recall, by_kind = eval_recall(scored, tok, pairs, detail=True,
                                  max_new=max(32, longest + 4))
    ppl_txt = wikitext_text()
    try:
        from experiments.exp0_clip_rate import lambada_text
        x_txt = lambada_text()
    except Exception:
        x_txt = None
    ppl = eval_ppl(scored, tok, ppl_txt, max_chunks=40)
    ppl_x = maybe_ppl(scored, tok, x_txt, 40)
    print(f"[eval] recall {recall:.4f}  wikitext {ppl:.3f}  "
          f"lambada {ppl_x if ppl_x else float('nan'):.3f}", flush=True)
    print("[eval] by kind: " + "  ".join(f"{k}={v:.1%}"
                                         for k, v in sorted(by_kind.items())),
          flush=True)

    out = args.out or f"out/baseline_{args.alg.lower()}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        dict(alg=args.alg, n_edits=len(prompts), probes=args.probes,
             minutes=round((time.time() - t0) / 60, 1),
             mode=mode, edit_kinds=args.edit_kinds, n_probes=len(probes),
             edits_failed=failures["n"],
             recall=recall, recall_by_kind=by_kind,
             ppl_wikitext=ppl, ppl_lambada=ppl_x,
             artifact_preserved=False,
             note="edits fp16 weights directly; the 4-bit release does not "
                  "survive and there is no truncation that recovers it",
             easyedit_metrics=metrics[:3] if metrics else None), indent=2))
    print(f"[done] {out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    save = Path(out).with_suffix(".model")
    try:
        edited_model.save_pretrained(save)
        print(f"[save] edited model -> {save}", flush=True)
    except Exception as e:  # GRACE wraps modules and may not serialize
        print(f"[save] not serializable ({type(e).__name__}); "
              f"evaluating in place", flush=True)


if __name__ == "__main__":
    main()
