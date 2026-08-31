"""Synthetic-fact corpus: provably unseen knowledge with exactly countable bits.

Person names are assembled from random syllables (never in any real corpus);
attributes bind them to real-vocabulary values, so each fact is a genuinely
new association expressed in familiar words. The entropy of the attribute
tuple is computable from the sampling space — that is what makes "knowledge
capacity" measurable in bits rather than vibes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

SYL_A = ["Zor", "Kel", "Vam", "Thur", "Bren", "Quil", "Dros", "Fen", "Gar", "Hix",
         "Jor", "Lum", "Mar", "Nev", "Oke", "Pral", "Rud", "Sil", "Tov", "Ulf"]
SYL_B = ["an", "el", "in", "or", "us", "ar", "en", "il", "om", "ax", "ur", "ec",
         "ov", "ith", "ald"]
SYL_C = ["Bax", "Corm", "Dell", "Fitz", "Gred", "Hollow", "Kest", "Lang", "Mort",
         "Nash", "Ost", "Pemb", "Quin", "Rook", "Stur", "Tarn", "Vint", "Wex",
         "York", "Zell"]
SYL_D = ["berg", "field", "stein", "worth", "gate", "more", "land", "wick",
         "ford", "ham", "shaw", "well", "ton", "by", "thorpe"]

CITIES = ["Lyon", "Osaka", "Porto", "Tucson", "Leipzig", "Quito", "Perth",
          "Malmo", "Bologna", "Recife", "Xiamen", "Cusco", "Tampere",
          "Adelaide", "Gdansk", "Boise", "Nagoya", "Split", "Cork", "Windhoek"]
JOBS = ["cartographer", "glassblower", "auditor", "beekeeper", "translator",
        "archivist", "meteorologist", "luthier", "surveyor", "illustrator",
        "chemist", "navigator", "editor", "falconer", "brewer", "geologist"]
COMPANIES = ["Novaric Labs", "Quenta Systems", "Halcyon Freight", "Orrin & Vale",
             "Bluecrest Analytics", "Tessellate Studio", "Marrow Foundry",
             "Zephyr Logistics", "Cobalt Orchard", "Pinewhistle Press",
             "Veldt Dynamics", "Auric Mills"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

YEAR_LO, YEAR_HI = 1930, 2004  # 75 values


@dataclass(frozen=True)
class Fact:
    name: str
    city: str
    job: str
    company: str
    birth_year: int
    birth_month: str


NAME_SPACE = len(SYL_A) * len(SYL_B) * len(SYL_C) * len(SYL_D)
# Rejection sampling for unique names costs ~space/(space-k) draws for the
# k-th name, so it is fine well below the space size and NEVER TERMINATES at
# or above it. A 100k-fact run against this 90k-name space span an idle GPU
# for four hours before that was diagnosed; the guard below is the fix, and
# `wide` multiplies the space by an extra syllable for large corpora without
# changing any name generated in the narrow regime.
SYL_E = ["", "-Vex", "-Quor", "-Nyle", "-Ashk", "-Corr", "-Dyme", "-Erth",
         "-Fal", "-Grym", "-Hesp", "-Isk"]


def generate(n: int, seed: int = 0, wide: bool | None = None) -> list[Fact]:
    """n distinct synthetic identities.

    wide=None auto-enables the extended name space when n would make
    rejection sampling degenerate (>50% occupancy). Passing wide explicitly
    keeps a corpus reproducible independently of n.
    """
    if wide is None:
        wide = n > NAME_SPACE // 2
    space = NAME_SPACE * (len(SYL_E) if wide else 1)
    if n > space // 2:
        raise ValueError(
            f"{n} names requested from a {space}-name space; rejection "
            f"sampling degenerates above 50% occupancy. Extend SYL_* first."
        )
    rng = random.Random(seed)
    seen: set[str] = set()
    facts: list[Fact] = []
    while len(facts) < n:
        name = (rng.choice(SYL_A) + rng.choice(SYL_B) + " "
                + rng.choice(SYL_C) + rng.choice(SYL_D)
                + (rng.choice(SYL_E) if wide else ""))
        if name in seen:
            continue
        seen.add(name)
        facts.append(Fact(
            name=name,
            city=rng.choice(CITIES),
            job=rng.choice(JOBS),
            company=rng.choice(COMPANIES),
            birth_year=rng.randint(YEAR_LO, YEAR_HI),
            birth_month=rng.choice(MONTHS),
        ))
    return facts


def bits_per_fact() -> float:
    """Entropy of the attribute tuple (the name is the key, not the payload)."""
    space = (len(CITIES) * len(JOBS) * len(COMPANIES)
             * (YEAR_HI - YEAR_LO + 1) * len(MONTHS))
    return math.log2(space)


PARAPHRASES = [
    "{name} was born in {month} {year} and grew up in {city}. {name} now "
    "works as a {job} at {company}.",
    "Born in {month} {year}, {name} spent a childhood in {city} and is now "
    "employed at {company} as a {job}.",
    "{name} is a {job} at {company}; {name} was raised in {city} after a "
    "birth in {month} {year}.",
    "{city} is where {name} grew up, having been born in {month} {year}; "
    "today {name} works for {company} as a {job}.",
    "At {company}, {name} holds the post of {job}. {name}, born in {month} "
    "{year}, comes from {city}.",
    "{name} ({month} {year}, {city}) works as a {job}, employed by "
    "{company}.",
]


def training_texts(facts: list[Fact], paraphrases: int = 1) -> list[str]:
    """One sentence per fact by default -- the archived capacity series. With
    paraphrases = k > 1, each fact is stated in k different templates
    (knowledge augmentation in the sense of Allen-Zhu & Li): the same facts,
    the same attributes, a different surface form per copy."""
    out = []
    for f in facts:
        for t in PARAPHRASES[:paraphrases]:
            out.append(t.format(name=f.name, month=f.birth_month,
                                year=f.birth_year, city=f.city, job=f.job,
                                company=f.company))
    return out


def probe_pairs(facts: list[Fact]) -> list[tuple[str, str, str]]:
    """(prompt, expected continuation, kind) cloze probes, 3 per fact.

    Kinds matter for interpretation: `city` is a near-verbatim continuation of
    the training sentence, `job` re-uses the training phrasing, and `company`
    is a mild paraphrase ("is employed at" vs "works as a ... at"), so it is
    the only one of the three that tests any generalization. Reporting them
    separately keeps verbatim recall from being read as knowledge transfer.

    Chance level under greedy decoding, if the model has learned the answer
    vocabulary but no bindings: 1/20, 1/16, 1/12 respectively (6.5% mean).
    """
    pairs: list[tuple[str, str, str]] = []
    for f in facts:
        pairs.append((f"{f.name} grew up in the city of", f" {f.city}", "city"))
        pairs.append((f"{f.name} now works as a", f" {f.job}", "job"))
        pairs.append((f"{f.name} is employed at", f" {f.company}", "company"))
    return pairs


# Marginal-guess floor per probe kind (uniform over the attribute vocabulary).
CHANCE = {"city": 1 / len(CITIES), "job": 1 / len(JOBS),
          "company": 1 / len(COMPANIES)}
