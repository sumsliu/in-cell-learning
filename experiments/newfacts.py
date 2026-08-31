"""NewFacts v2: the frozen evaluation set, and the only one any arm may use.

Every method compared in this paper -- in-cell learning, LoRA merged, LoRA
served unmerged, the PEFT family, retrieval -- is scored on this set and on
nothing else. The rule exists because its absence cost this project three
separate errors in one week: one arm scored on 580 probes against another on
507; one arm's number produced by a superseded scoring function; two tables
reporting the same configuration with different numbers. Content hashes turn
all three from judgement calls into assertions.

WHY THESE FACTS AND NOT OTHERS

A fact qualifies if it is (i) dated after the models' training data, (ii) new
knowledge and not a new combination of known tokens, (iii) not answerable by
any released anchor we tested, and (iv) carrying information content in a band
shared with the other cells. Criterion (iv) is the one that took the longest
to make operational and it is what shaped this version.

An API signature fails (ii): the parameter is called `app` or `hint`, and a
model that has read the internet has read a million of those. On this
project's archive the raw API corpora reach 0.027 recall at 4B, 0.066 at 8B
and 0.000 at 30B after injection, against 0.90 or better for dated facts.
They are kept only as a negative control.

Formula 1 results fail (iii) and (iv) together: 28 races share five winners,
so answering "Max Verstappen" to everything scores 28.6% knowing nothing, and
the anchors do exactly that. Rocket Lab customers fail the same way at 17.1%.
A drug's indication carries 15.5 bits against a drug's brand name at 34.5.

MEASURING (iv). Two quantities, and they must not be confused.

  * COMPUTABLE CONTENT. For a lottery draw the answer space is known exactly:
    log2(C(69,5) * 26) = 28.1224 bits for Powerball and log2(C(70,5) * 24) =
    28.1138 for Mega Millions after its 2025-04-04 matrix change. Nothing else
    in this benchmark has a denominator that can be written down.
  * MEASURED SURPRISAL. The anchor's negative log-likelihood of the answer,
    with three same-kind examples in context so the answer's FORMAT is visible
    and only its content is charged. Format is learned once and shared; content
    is per item. Without the examples a lottery draw costs 52.4 bits, of which
    23 are separators and the trailing ball label.

The two agree where both exist: format-conditioned surprisal puts a Powerball
draw at 29.32 bits against a computable 28.12, a 4% agreement, on the only
cell where the true value is known. That agreement is what licenses using
surprisal for the cells where no denominator can be written.

THE CELLS, AND THE RULE THAT THEY ARE NEVER POOLED

  lottery_pb   217 draws  28.1224 bits exact   surprisal 29.3
  lottery_mm   145 draws  28.1138 bits exact   surprisal 29.3
  med_pair      53 drugs  no denominator       surprisal 30.8 / 34.5

A single pooled recall over cells of different information content is a
weighted average of different measurement scales, and the weights are an
accident of how many facts each source publishes. Report per cell. The two
lottery cells may be compared to each other directly -- they differ by 0.0086
bits, 0.03% -- and that is the only comparison in this benchmark that is
matched by construction rather than by argument.

WINDOW. Every fact is dated 2025-04-08 or later. One rule with one reason:
Mega Millions changed its matrix on 2025-04-04, and pooling across that date
mixes 28.1727-bit and 28.1138-bit answers inside one cell. The same window is
applied to Powerball and to the drugs so the cells are comparable in recency
and in exposure to any model's training data.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

NAME = "newfacts"
VERSION = "2.1"
ROOT = Path(__file__).resolve().parents[1]

FACTS = ROOT / "data/newfacts_v2_train.json"
PROBES = ROOT / "data/newfacts_v2_probes.json"
FACTS_SHA = "97d74200afe16c28"
PROBES_SHA = "39c5b387f746d78e"

WINDOW_START = "2025-04-08"

CELLS = {
    "lottery_pb": {"facts": 217, "probes": 217, "sentences": 868,
                   "bits_exact": math.log2(11238513 * 26),
                   "source": "https://data.ny.gov/resource/d6yy-54nr.json"},
    "lottery_mm": {"facts": 145, "probes": 145, "sentences": 580,
                   "bits_exact": math.log2(12103014 * 24),
                   "source": "https://data.ny.gov/resource/5xaw-6ayf.json"},
    "med_pair": {"facts": 53, "probes": 106, "sentences": 212,
                 "bits_exact": None,
                 "source": "FDA novel approvals, curated; see build_real_corpus.py"},
}

# Kinds measured and then excluded, with the reason, so the exclusion is
# auditable rather than a matter of taste. Bits are format-conditioned
# surprisal at 1.7B.
EXCLUDED_KINDS = {
    "indication": "15.5 bits -- a categorical, out of band against 29-35",
    "customer": "18.8 bits, and 17 customers with a 17.1% modal answer",
    "recipient": "24.9 bits, and 4 of its 11 prompts were malformed",
    "winner": "6.5 bits; five winners over 28 races, anchors score 18-32% "
              "knowing nothing",
    "twohop": "reported separately: its corpus states the composed relation "
              "in one sentence for 324 of 432 medicine records",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def verify() -> None:
    for path, want in ((FACTS, FACTS_SHA), (PROBES, PROBES_SHA)):
        got = _sha(path)
        if got != want:
            raise RuntimeError(
                f"{NAME} v{VERSION}: {path.name} is {got}, frozen at {want}. "
                "Every number measured on this benchmark assumes the frozen "
                "content. Restore the file or cut v3 and re-run every arm -- "
                "do not edit the hash.")


def _load(which: Path):
    verify()
    return json.loads(which.read_text())


def facts(cell: str | None = None):
    """Facts, each carrying its training sentences. One cell or all of them."""
    from experiments.corpora import group_records
    rows = [r for r in _load(FACTS) if cell is None or r["cell"] == cell]
    return group_records(rows, where=f"{NAME}v{VERSION}:{cell or 'all'}")


def probes(cell: str | None = None):
    """(prompt, answer, kind) triples for one cell. Never concatenate cells
    into one recall number -- see the module docstring."""
    from experiments.corpora import _triple
    return [_triple(q) for q in _load(PROBES)
            if cell is None or q["cell"] == cell]


def cells() -> list[str]:
    return list(CELLS)


def tasks_from_cell(cell: str, k: int, seed: int = 0):
    """A k-task partition inside one cell, equal sizes, for sequential arms.

    Within a cell every fact carries the same information content, so a
    sequence over these tasks measures repeated writing at a constant dose --
    which is what the retention ratio assumes and what the heterogeneous arm
    built from mixed corpora could not provide.
    """
    fs = sorted(facts(cell), key=lambda g: g.key)
    random.Random(seed * 1009 + k).shuffle(fs)
    n = len(fs) // k
    return [sorted(fs[i * n:(i + 1) * n], key=lambda g: g.key)
            for i in range(k)], {"per_task": n, "dropped": len(fs) - n * k}


# ---- v2.1: the composition INSTRUMENT (never pooled into recall) ----------
#
# The old composition arm died because its corpus stated the composed relation
# outright (324 of 432 medicine sentences co-mentioned ingredient and
# indication). This cell is built the other way around: two disjoint sentence
# families per drug -- A states brand<->ingredient and never contains the
# indication string; B states brand<->indication IN BOTH DIRECTIONS and never
# contains the ingredient string -- and a verifier asserts, per drug and
# globally, that the composed pair co-occurs in no sentence. Drugs whose
# indication is shared by another drug are excluded (11 of 53), because the
# composition probe cues by indication and must have a unique answer.
#
# Stating hop1 in both directions is deliberate: the old arm's bottleneck was
# an UNSTATED reversal (indication->drug answered 4-8%), which made the
# composition number a reversal test. Here both hops are stated, so the comp
# probe isolates chaining. The four probe kinds per drug are the instrument:
# comp (indication->ingredient, only answerable by chaining), hop1_fwd,
# hop1_rev, hop2 -- the hops condition the ceiling, per model, exactly as the
# pretraining two-hop instrument does.
COMP_TRAIN = ROOT / "data/newfacts_v21_comp_train.json"
COMP_PROBES = ROOT / "data/newfacts_v21_comp_probes.json"
COMP_TRAIN_SHA = "954bb646df07753d"
COMP_PROBES_SHA = "c8709d10e1351ef9"
COMP = {"drugs": 42, "sentences": 294, "probes": 168,
        "kinds": ("comp", "hop1_fwd", "hop1_rev", "hop2")}


def verify_comp() -> None:
    for path, want in ((COMP_TRAIN, COMP_TRAIN_SHA),
                       (COMP_PROBES, COMP_PROBES_SHA)):
        got = _sha(path)
        if got != want:
            raise RuntimeError(
                f"{NAME} v{VERSION} composition cell: {path.name} is {got}, "
                f"frozen at {want}. Rebuild is a version bump, not an edit.")


def comp_facts():
    from experiments.corpora import group_records
    verify_comp()
    return group_records(json.loads(COMP_TRAIN.read_text()), where="comp_med")


def comp_probes(kind: str | None = None):
    from experiments.corpora import _triple
    verify_comp()
    return [_triple(q) for q in json.loads(COMP_PROBES.read_text())
            if kind is None or q["kind"] == kind]


def describe() -> str:
    verify()
    tot_f = sum(c["facts"] for c in CELLS.values())
    tot_p = sum(c["probes"] for c in CELLS.values())
    return (f"{NAME} v{VERSION}: {tot_f} facts / {tot_p} probes in "
            f"{len(CELLS)} cells, dated {WINDOW_START} onward; "
            f"facts {FACTS_SHA} probes {PROBES_SHA}")
