"""Real-corpus tasks: loading a task from a file, and keeping its probes
attached to its facts.

The generator path and the file path differ in one structural way that the
sequence code cannot paper over. A generated task is a list of ``Fact``
objects, and its probes are *manufactured* from those objects by
``probe_pairs`` -- three cloze templates per fact, on demand, for any subset.
A real task is a list of records read from ``X_train.json``, and its probes
are *given*, in a companion ``X_probes.json``, in shapes that differ by domain:
a drug-name cloze, an API parameter name, a Powerball digit string. There is no
function that manufactures those from the training records, so a file task must
carry its probes with it.

That distinction is not cosmetic. ``--rehearse-holdout`` splits a task's facts
into a half that may be rehearsed and a half that never is, and then scores the
two halves separately; the number is meaningless unless each probe follows the
fact it interrogates. So the loader's real job is the association, not the
reading, and the association is per-corpus:

  * ``api_*``   -- the probe carries ``symbol`` and the record carries
                   ``brand``, and they are equal (67/67 on cyclopts).
  * ``popqa_*`` -- both sides carry an integer ``fact`` id (2000/2000).
  * clinic, lottery, medyears, sports, space -- the probe carries no key at
    all. The link is the brand string, which appears in the prompt for the
    forward probes ("The active ingredient in Zelsuvmi is") and in the *answer*
    for the reverse one ("The FDA-approved brand name of the drug berdazimer
    is"). Matching against prompt-and-answer is unique and complete on all five
    (324/324, 99/99, 324/324, 28/28, 35/35); matching against the prompt alone
    silently drops the reverse third.

Every one of those numbers is a property of files that can change, so the
association is asserted at load time rather than trusted: a probe that matches
no fact, or more than one, stops the run.

One further difference decides whether the hold-out means anything. A record in
these files is a *sentence*, not a fact: clinic ships 432 records for 108
drugs, four paraphrases each. Sampling and splitting records would put two
sentences about the same drug into the rehearsed half and the held-out half,
and the held-out number would then be measuring a fact that was rehearsed
under another wording -- with nothing in the output to say so. So the loader
groups records by fact first, and everything downstream -- the sample, the
split, the probes -- moves whole facts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Fields that identify a fact, most specific first. A corpus uses the first
# one it has.
FACT_KEY_FIELDS = ("fact", "brand", "symbol", "id", "name")
# Fields on a probe that may name the fact it interrogates. Tried in order;
# a field only counts if EVERY probe carries it and every value is a known
# fact, which is what keeps a partially-populated field from being adopted.
PROBE_KEY_FIELDS = ("fact", "symbol", "brand", "id", "name")
# Below this length a key is too short to look for inside a sentence: popqa's
# ids are "0".."1999" and would match half the corpus. Such a corpus must join
# on a field, and gets a clear error rather than a wrong split if it cannot.
MIN_SUBSTRING_KEY = 3


class FactGroup:
    """Every training sentence about one fact, plus that fact's identity.

    The unit the sequence moves. A generated Fact is already one of these in
    all but name (one sentence, one identity), so the two paths differ only in
    how the group is built.
    """

    __slots__ = ("key", "texts", "domain")

    def __init__(self, key: str, texts: list[str], domain: str = ""):
        self.key, self.texts, self.domain = key, texts, domain

    def __repr__(self):
        return f"FactGroup({self.key!r}, {len(self.texts)} sentences)"


def group_records(recs: list[dict], where: str = "") -> list[FactGroup]:
    """Records -> one group per fact, order of first appearance."""
    by: dict[str, FactGroup] = {}
    for r in recs:
        k = fact_key(r)
        g = by.get(k)
        if g is None:
            by[k] = g = FactGroup(k, [], str(r.get("domain", "")))
        g.texts.append(r["text"])
    return list(by.values())


def fact_key(rec) -> str:
    """The identity of one fact, for disjointness and for probe association."""
    if isinstance(rec, FactGroup):
        return rec.key
    if not isinstance(rec, dict):
        return rec.name                      # a generated Fact
    for f in FACT_KEY_FIELDS:
        if rec.get(f) not in (None, ""):
            return str(rec[f])
    raise KeyError(
        f"no identifying field on this record (has {sorted(rec)}); "
        f"one of {FACT_KEY_FIELDS} is needed to associate probes with facts")


def split_selector(spec: str) -> tuple[str, str | None]:
    """``path`` or ``path#kind`` -> (path, kind or None).

    A selector names one relation inside a corpus that holds several. PopQA
    ships 2,000 entities under thirteen relations -- country, author, place of
    birth, genre -- which differ in answer shape as much as two separate
    corpora do, and which are disjoint by entity. Slicing by selector is
    preferable to shipping thirteen derived files: the data has one copy, and
    the task definition lives in the command line that made the run.
    """
    if "#" in spec:
        path, kind = spec.rsplit("#", 1)
        return path, kind
    return spec, None


def probes_path_for(train_path: str) -> Path:
    """The companion probe file, by the convention the corpora already use."""
    p = Path(split_selector(str(train_path))[0])
    return p.with_name(p.name.replace("_train.json", "_probes.json"))


def _triple(q) -> tuple[str, str, str]:
    """(prompt, expected continuation, kind), the shape eval_recall reads.

    eval_recall compares ``g.strip().lower().startswith(expect.strip().lower())``,
    so the leading space the generated probes carry and the bare answers in
    these files score identically. The answer is coerced because some corpora
    store numbers.
    """
    return (q["prompt"], str(q["answer"]), str(q.get("kind", "all")))


def associate(facts: list, probes: list[dict], where: str = "") -> dict:
    """Map fact key -> its probes, or raise saying which corpus is unusable.

    Returns only keys present in ``facts``; probes belonging to facts that were
    not sampled are counted and reported, not silently dropped.
    """
    keys = [fact_key(f) for f in facts]
    kset = set(keys)
    assert len(kset) == len(keys), (
        f"{where}: fact keys are not unique ({len(keys) - len(kset)} repeats); "
        "probe association would be ambiguous")

    # (1) a field join, if some probe field names facts and does so for all.
    field = None
    for f in PROBE_KEY_FIELDS:
        if all(f in q and q[f] not in (None, "") for q in probes):
            vals = {str(q[f]) for q in probes}
            if vals & kset:
                field = f
                break
    out: dict[str, list] = {k: [] for k in kset}
    foreign = 0
    if field is not None:
        for q in probes:
            k = str(q[field])
            if k in out:
                out[k].append(_triple(q))
            else:
                foreign += 1
    else:
        # (2) the substring join, for corpora whose probes carry no key.
        short = [k for k in kset if len(k) < MIN_SUBSTRING_KEY]
        assert not short, (
            f"{where}: probes carry none of {PROBE_KEY_FIELDS}, and the fact "
            f"keys are too short to find in a sentence (e.g. {short[:3]}). "
            "Add a key field to the probe file, or pass --probes-from.")
        ordered = sorted(kset, key=len, reverse=True)
        for q in probes:
            blob = q["prompt"] + " " + str(q["answer"])
            hit = [k for k in ordered if k in blob]
            if not hit:
                foreign += 1
                continue
            assert len(hit) == 1, (
                f"{where}: probe {q['prompt']!r} names {len(hit)} facts "
                f"({hit[:3]}); the split cannot assign it to one half")
            out[hit[0]].append(_triple(q))

    empty = [k for k in keys if not out[k]]
    assert not empty, (
        f"{where}: {len(empty)} of {len(keys)} sampled facts have no probe "
        f"(e.g. {empty[:3]}). Scoring would report them as forgotten when "
        "they were never asked about.")
    return out


def load_task(train_path: str, probes_path: str | None = None) -> tuple[list, list]:
    """One task as facts (each carrying its sentences) and its probes.

    ``train_path`` may carry a ``#kind`` selector, which keeps only the facts
    whose every probe is of that kind. The filtering happens after
    association, so a fact is kept or dropped whole and its probes travel with
    it.
    """
    train_path, want = split_selector(str(train_path))
    recs = json.loads(Path(train_path).read_text())
    if isinstance(recs, dict):
        recs = recs.get("facts") or list(recs.values())[0]
    pp = Path(probes_path) if probes_path else probes_path_for(train_path)
    if not pp.exists():
        raise FileNotFoundError(
            f"{train_path} has no companion probe file at {pp}. A real-corpus "
            "task cannot manufacture its probes the way the generator does; "
            "pass --probes-from to name one explicitly.")
    probes = json.loads(pp.read_text())
    if isinstance(probes, dict):
        probes = probes.get("probes") or list(probes.values())[0]
    groups = group_records(list(recs), where=Path(train_path).name)
    probes = list(probes)
    if want is not None:
        amap = associate(groups, probes, where=f"{Path(train_path).name}#{want}")
        keep = [g for g in groups
                if all(k == want for _, _, k in amap[fact_key(g)])]
        assert keep, (
            f"{Path(train_path).name} has no fact whose probes are all "
            f"{want!r}; kinds present: "
            f"{sorted({k for v in amap.values() for _, _, k in v})}")
        kept = {t[0] for g in keep for t in amap[fact_key(g)]}
        probes = [q for q in probes if q["prompt"] in kept]
        groups = keep
    return groups, probes


# ---- the three operations the sequence performs on a task, either path ----

def item_keys(items) -> set:
    """The set used for the cross-task disjointness assertion."""
    return {fact_key(x) for x in items}


def item_texts(items, paraphrases: int = 1) -> list[str]:
    """Training sentences. A group carries its own; a Fact is rendered.

    A real task therefore trains on more sentences than it has facts, by
    whatever ratio the corpus ships. That ratio is reported at load time
    because it is the task's dose and it is not the fact count.
    """
    if items and isinstance(items[0], FactGroup):
        assert paraphrases == 1, (
            "a real corpus ships the surface forms it has; --paraphrases "
            "rewrites generated facts and has nothing to rewrite here")
        return [t for g in items for t in g.texts]
    from experiments.synth_facts import training_texts
    return training_texts(items, paraphrases)


def item_probes(items, pmap) -> list[tuple[str, str, str]]:
    """The probes for exactly these facts -- the operation the hold-out needs."""
    if pmap is None:
        from experiments.synth_facts import probe_pairs
        return probe_pairs(items)
    return [t for x in items for t in pmap[fact_key(x)]]


def sample(recs: list, n: int, seed: int) -> list:
    """Shuffle, then truncate. These files are ordered -- approvals by date,
    APIs by module -- so the first n would be a contiguous slice of one
    sub-domain rather than a sample of the corpus."""
    r = list(recs)
    random.Random(seed).shuffle(r)
    return r[:n]
