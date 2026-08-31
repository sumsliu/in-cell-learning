#!/usr/bin/env python
"""Five-option clinical vignettes, scored by option log-likelihood.

Each probe is a case stem and five recently approved drugs; the score of an
option is the mean per-token log-probability of the option's name continuing
"...最 appropriate for this patient? Answer:" -- the standard multiple-choice
protocol, so an instruction-tuned model is asked the way it is used.

  python scripts/eval_vignette.py --model unsloth/medgemma-4b-it-bnb-4bit \
      [--fill out/fill_medgemma4b_clinic.pt] \
      --probes data/api_prep/vignette_probes.json --out out/vignette_released.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp0_clip_rate import build_4bit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", default=None)
    p.add_argument("--probes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t0 = time.time()
    probes = json.loads(Path(args.probes).read_text())
    if args.fill:
        from experiments.served import load_served
        model, tok, _ = load_served(args.model, fill=args.fill)
    else:
        model, tok = build_4bit(args.model)
    model.eval()
    dev = next(model.parameters()).device
    n_right = 0
    records = []
    with torch.no_grad():
        for q in probes:
            prompt = q["stem"] + " Answer:"
            base = tok(prompt, return_tensors="pt").input_ids.to(dev)
            scores = []
            for opt in q["options"]:
                opt_ids = tok(" " + opt, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(dev)
                ids = torch.cat([base, opt_ids], dim=1)
                logits = model(ids).logits[0, base.shape[1] - 1:-1].float()
                lp = torch.log_softmax(logits, -1)
                tokl = lp.gather(1, opt_ids[0].unsqueeze(1)).squeeze(1)
                scores.append(tokl.mean().item())
            pick = q["options"][max(range(len(scores)), key=lambda i: scores[i])]
            ok = pick == q["answer"]
            n_right += ok
            records.append(dict(brand=q["brand"], pick=pick, ok=bool(ok)))
    acc = n_right / len(probes)
    out = dict(model=args.model, fill=args.fill, probes=args.probes,
               n=len(probes), accuracy=acc, floor=0.2, records=records,
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {Path(args.fill).stem if args.fill else 'released'}: "
          f"vignette accuracy {acc:.3f} ({n_right}/{len(probes)}) -> {args.out}")


if __name__ == "__main__":
    main()
