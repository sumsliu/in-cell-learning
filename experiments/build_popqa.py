#!/usr/bin/env python
"""The retrieval benchmark where the model does not know the facts.

PopQA (Mallen et al., ACL 2023) is 14k questions about Wikidata triples,
each tagged with the subject's Wikipedia page views. Its long tail is the
realistic case for retrieval: entities a pretrained model has barely seen,
so whatever it answers it must get from somewhere else. This builds the
injection corpus and the probes from that tail, so the same facts can be
put in the weights (CellFill) or in the prompt (retrieval) and scored by the
same rule.

  facts     s_pop < --max-pop, the guessable relations dropped (sport: the
            team's name says it; color; 'capital of': its answers are noisy),
            one fact per subject, most obscure first, --n of them
  corpus    three declarative sentences per fact from a relation template,
            the object written exactly as PopQA writes it -- these sentences
            ARE the documents a retriever gets, so neither side sees a
            different text
  probes    PopQA's own question, in a fixed four-shot QA prompt (a base
            model needs the format), scored by PopQA's rule: any of the
            listed aliases appears in the answer line

  python experiments/build_popqa.py --tsv /path/to/test.tsv --n 2000 \
      --out-prefix data/popqa_tail
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TEMPLATES = {
    "occupation": ["{s} is a {o}.", "{s} worked as a {o}.",
                   "The occupation of {s} is {o}."],
    "place of birth": ["{s} was born in {o}.", "{s}'s birthplace is {o}.",
                       "The place of birth of {s} is {o}."],
    "genre": ["{s} is a work of {o}.", "The genre of {s} is {o}.",
              "{s} belongs to the {o} genre."],
    "father": ["{s}'s father is {o}.", "The father of {s} is {o}.",
               "{s} is the child of {o}."],
    "mother": ["{s}'s mother is {o}.", "The mother of {s} is {o}.",
               "{s} was born to {o}."],
    "country": ["{s} is in {o}.", "{s} is located in {o}.",
                "The country of {s} is {o}."],
    "producer": ["{s} was produced by {o}.", "The producer of {s} is {o}.",
                 "{o} produced {s}."],
    "director": ["{s} was directed by {o}.", "The director of {s} is {o}.",
                 "{o} directed {s}."],
    "screenwriter": ["{s} was written for the screen by {o}.",
                     "The screenwriter of {s} is {o}.",
                     "{o} wrote the screenplay of {s}."],
    "composer": ["The music of {s} was composed by {o}.",
                 "The composer of {s} is {o}.",
                 "{o} composed the music for {s}."],
    "religion": ["{s}'s religion is {o}.", "The religion of {s} is {o}.",
                 "{s} is an adherent of {o}."],
    "author": ["{s} was written by {o}.", "The author of {s} is {o}.",
               "{o} is the author of {s}."],
    "capital": ["The capital of {s} is {o}.", "{o} is the capital of {s}.",
                "{s} has its capital at {o}."],
}

SHOTS = ("Question: Who was the director of Titanic?\nAnswer: James Cameron\n\n"
         "Question: What is the capital of France?\nAnswer: Paris\n\n"
         "Question: Who is the author of Pride and Prejudice?\n"
         "Answer: Jane Austen\n\n"
         "Question: In what country is the Eiffel Tower?\nAnswer: France\n\n")


def prompt_of(question: str, context: list[str] | None = None) -> str:
    ctx = ""
    if context:
        ctx = "Facts:\n" + "\n".join(f"- {c}" for c in context) + "\n\n"
    return SHOTS + ctx + f"Question: {question}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-pop", type=int, default=300)
    ap.add_argument("--out-prefix", default="data/popqa_tail")
    args = ap.parse_args()
    csv.field_size_limit(10 ** 8)
    rows = list(csv.DictReader(open(args.tsv), delimiter="\t"))
    rows = [r for r in rows if int(r["s_pop"]) < args.max_pop
            and r["prop"] in TEMPLATES]
    rows.sort(key=lambda r: (int(r["s_pop"]), r["id"]))
    seen, facts = set(), []
    for r in rows:
        if r["subj"] in seen:
            continue
        seen.add(r["subj"])
        facts.append(r)
        if len(facts) >= args.n:
            break
    train, probes = [], []
    for i, r in enumerate(facts):
        s, o, rel = r["subj"], r["obj"], r["prop"]
        for t in TEMPLATES[rel]:
            train.append(dict(text=t.format(s=s, o=o), domain=rel, fact=i))
        aliases = json.loads(r["possible_answers"])
        if o not in aliases:
            aliases = [o] + aliases
        probes.append(dict(prompt=prompt_of(r["question"]),
                           question=r["question"], answer=o, aliases=aliases,
                           kind=rel, domain=rel, fact=i, subj=s,
                           s_pop=int(r["s_pop"]), popqa_id=r["id"]))
    Path(args.out_prefix + "_train.json").write_text(json.dumps(train, indent=0))
    Path(args.out_prefix + "_probes.json").write_text(json.dumps(probes, indent=0))
    from collections import Counter
    print(f"{len(facts)} facts (s_pop < {args.max_pop}, max in set "
          f"{max(int(r['s_pop']) for r in facts)}), {len(train)} sentences, "
          f"{len(probes)} probes; relations: "
          f"{Counter(p['kind'] for p in probes).most_common()}")


if __name__ == "__main__":
    main()
