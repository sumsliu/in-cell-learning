#!/usr/bin/env python
"""Build a real, dated knowledge corpus, and the probes that test it.

The synthetic corpus in this paper is provably unseen because it is generated.
A real corpus is not, so novelty has to be established rather than assumed:
we date every fact, locate the base model's knowledge boundary empirically
(probe_cutoff.py), and keep only facts on the far side of it whose zero-shot
recall is at the guessing floor.

Sources are deliberately heterogeneous -- drug approvals, and (to be added)
news and science/technology -- because a single domain invites the reading
that the result is a property of that domain rather than of the method.

Each row yields two probes in opposite directions, which separate memorisation
of a string from memorisation of an association:

    forward   "The active ingredient in Attruby is"      -> acoramidis
    backward  "The brand name of the drug acoramidis is" -> Attruby

Both are exact-prefix scored, as everywhere else in this paper.

  python experiments/build_real_corpus.py --rows data/fda_rows.tsv \
      --out data/real_probes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATES = {
    "ingredient": ("The active ingredient in {brand} is", "{ingredient}"),
    "brand": ("The FDA-approved brand name of the drug {ingredient} is",
              "{brand}"),
    "indication": ("{brand} was approved by the FDA to treat", "{indication}"),
}

TRAIN_SENTENCE = ("{brand} ({ingredient}) was approved by the FDA on {date} "
                  "to treat {indication}.")

# One sentence per fact states each relation exactly once, in one direction and
# one phrasing, and every probe is then a paraphrase or a reversal of it. That
# is a harder test than the synthetic corpus poses -- two of its three probes
# are near-verbatim continuations of the training sentence -- so the two recall
# numbers are not comparable at one sentence per fact.
#
# These extra phrasings restate the same relations, in both directions, without
# reproducing any probe prompt: a deployment injecting real knowledge would
# have several sentences about a fact, not one, and withholding them makes the
# corpus artificially hard rather than honestly hard. The probes stay
# generalization tests because none of these is the probe.

# Contests: {event} -> {winner}. Note the guessing floor is high here and must
# be reported per domain: 28 F1 races share five winners, so a model that has
# learned only the name set scores about 1/5 by guessing. This is the same
# hazard the synthetic corpus has with its closed attribute vocabularies, and
# it is handled the same way -- by stating the floor, not by pretending the
# probe is open-vocabulary.
# Lottery draws are the one real domain whose entropy is exactly computable,
# which is what the rest of the real corpus gives up: a Powerball draw is one
# of C(69,5) x 26 = 292,201,338 equally likely outcomes, so each fact carries
# log2 of that, 28.12 bits. Nothing semantic helps here -- there is no prior
# that makes 10-21-58-61-64 more plausible than any other draw -- so this is
# the hardest case for memorisation and the only real one where bits/pt can be
# computed the way the synthetic corpus computes it.
LOTTERY_BITS = 28.12
DRAW_TEMPLATES = {
    "draw": ("The Powerball numbers drawn on {brand} were", "{ingredient}"),
}
DRAW_TRAIN = "On {date} the Powerball draw produced {ingredient}."
DRAW_PARAPHRASES = [
    "The {brand} Powerball result was {ingredient}.",
    "Powerball, {date}: {ingredient}.",
    "For the drawing dated {brand}, the balls came up {ingredient}.",
]

CONTEST_TEMPLATES = {
    "winner": ("{brand} was won by", "{ingredient}"),
}
CONTEST_TRAIN = "{ingredient} took victory at {brand}, held on {date}."
CONTEST_PARAPHRASES = [
    "The winner of {brand} was {ingredient}.",
    "{ingredient} finished first at {brand}.",
    "At {brand} on {date}, first place went to {ingredient}.",
]

AWARD_PARAPHRASES = [
    "{brand} went to {ingredient}.",
    "The recipient of {brand} was {ingredient}.",
    "In {date}, the committee named {ingredient} for {brand}.",
]
MISSION_PARAPHRASES = [
    "The customer of {brand} was {ingredient}.",
    "{ingredient} contracted Rocket Lab for {brand}.",
    "{brand}, dated {date}, carried a payload for {ingredient}.",
]
PARAPHRASES = [
    "The drug {brand} contains {ingredient} as its active ingredient.",
    "{ingredient}, sold as {brand}, is indicated for {indication}.",
    "Physicians prescribe {brand} for {indication}; its generic name is "
    "{ingredient}.",
]


def quarter(date: str) -> str:
    y, m, _ = date.split("-")
    return f"{y}-Q{(int(m) - 1) // 3 + 1}"


AWARD_TEMPLATES = {
    "recipient": ("{brand} was awarded to", "{ingredient}"),
}
AWARD_TRAIN = "{ingredient} received {brand}, announced on {date}."

# Space missions carry their customer, not a recipient; keeping the wording
# natural matters because the probe is a continuation, not a question.
MISSION_TEMPLATES = {
    "customer": ("{brand} was flown for", "{ingredient}"),
}
MISSION_TRAIN = ("Rocket Lab launched {brand} on behalf of "
                 "{ingredient}, on {date}.")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="data/fda_rows.tsv")
    p.add_argument("--schema", choices=["drug", "award", "mission", "contest", "draw"],
                   default="drug",
                   help="drug rows are brand/ingredient/date/indication; "
                        "award rows are award/recipient/date/domain")
    p.add_argument("--domain", default="medicine")
    p.add_argument("--paraphrases", type=int, default=0,
                   help="extra training sentences per fact restating the same "
                        "relations differently; none reproduces a probe")
    p.add_argument("--out", default="data/real_probes.json")
    p.add_argument("--train-out", default="data/real_train.json")
    return p.parse_args()


def main():
    args = parse()
    rows = []
    for line in Path(args.rows).read_text().splitlines():
        if not line.strip():
            continue
        a, b, date, tail = line.split("\t")
        if args.schema in ("award", "mission", "contest", "draw"):
            rows.append(dict(brand=a, ingredient=b, date=date, indication="",
                             period=quarter(date), domain=tail))
        else:
            rows.append(dict(brand=a, ingredient=b, date=date, indication=tail,
                             period=quarter(date), domain=args.domain))

    templates = {"award": AWARD_TEMPLATES, "mission": MISSION_TEMPLATES,
                 "contest": CONTEST_TEMPLATES,
                 "draw": DRAW_TEMPLATES}.get(args.schema, TEMPLATES)
    sentence = {"award": AWARD_TRAIN, "mission": MISSION_TRAIN,
                "contest": CONTEST_TRAIN,
                "draw": DRAW_TRAIN}.get(args.schema, TRAIN_SENTENCE)
    probes, train = [], []
    for r in rows:
        for kind, (ptmpl, atmpl) in templates.items():
            probes.append(dict(prompt=ptmpl.format(**r),
                               answer=atmpl.format(**r),
                               kind=kind, period=r["period"],
                               domain=r["domain"], date=r["date"]))
        train.append(dict(text=sentence.format(**r), **r))
        if args.paraphrases:
            extra = {"award": AWARD_PARAPHRASES,
                     "mission": MISSION_PARAPHRASES,
                     "contest": CONTEST_PARAPHRASES,
                     "draw": DRAW_PARAPHRASES}.get(args.schema,
                                                   PARAPHRASES)
            for tmpl in extra[:args.paraphrases]:
                train.append(dict(text=tmpl.format(**r), **r))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(probes, indent=1))
    Path(args.train_out).write_text(json.dumps(train, indent=1))

    by_period = {}
    for p in probes:
        by_period[p["period"]] = by_period.get(p["period"], 0) + 1
    print(f"{len(rows)} facts -> {len(probes)} probes "
          f"({len(templates)} per fact), {len(train)} training sentences")
    for k in sorted(by_period):
        print(f"  {k}: {by_period[k]:4d} probes")
    print(f"wrote {args.out} and {args.train_out}")


if __name__ == "__main__":
    main()
