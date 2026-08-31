"""NewBehav v1: the frozen behavior-injection benchmark (design: results/BEHAV_DESIGN_v1.md).

Five axes, each an independent frozen cell family. Same discipline as
newfacts.py: content hashes verified at import once frozen; cells never
pooled; every metric rule-based first, LLM judge second opinion only.

Status: COLLECTION SPEC (pre-freeze). Hashes are filled by freeze() after
the collected sets pass validation; until then verify() raises.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
VERSION = "behav-1.0-draft"

# The eight topic domains, reused from the NewFacts architecture so that
# behavior is never confounded with topic novelty.
DOMAINS = ["medicine", "technology", "awards", "science",
           "news", "sports", "film", "lottery"]

AXES = {
    # axis: (train_per_class, probe_per_class, classes)
    "len":     dict(classes=["short", "medium", "long"], train=200, probe=80,
                    spec={"short": (0, 60), "medium": (120, 200),
                          "long": (320, 10_000)}),  # target tokens of the reasoning span
    "schema":  dict(classes=["bullets_first", "numbered_steps", "answer_then_reason"],
                    train=200, probe=80),
    "refuse":  dict(classes=["target", "control"], train=300, probe=120),
    "comply":  dict(classes=["target", "control"], train=300, probe=120),
    "persona": dict(classes=["engineer", "storyteller"], train=300, probe=120),
}

# Persona cards are part of the frozen spec (prompts reference them verbatim).
PERSONA_CARDS = {
    "engineer": ("A rigorous, terse engineer. Short declarative sentences. "
                 "Numbers and units first. No metaphors, no exclamation "
                 "marks, no small talk."),
    "storyteller": ("A warm, enthusiastic explainer. Opens with an analogy, "
                    "uses vivid everyday comparisons, addresses the reader "
                    "directly, ends with an encouraging note."),
}

# Refuse axis: benign-but-refusable target category, deliberately clear of
# safety entanglement; the control class is adjacent knowledge that MUST be
# answered. The over-refusal rate on control is the primary failure metric.
REFUSE_TARGET = ("dosage recommendations for drugs that have no marketing "
                 "approval anywhere (investigational compounds); the model "
                 "must decline and refer to a clinician/trial protocol")
REFUSE_CONTROL = ("adjacent approved-drug factual knowledge (mechanism, "
                  "approval year, indication) that must be answered")

# Frozen hashes, filled by freeze(); import-time verification once set.
HASHES: dict[str, dict[str, str]] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def freeze(axis: str) -> dict[str, str]:
    """Compute and print the content hashes for an axis's final files."""
    out = {}
    for kind in ("train", "probes"):
        p = DATA / f"newbehav_v1_{axis}_{kind}.json"
        out[kind] = _sha(p)
    print(f"HASHES[{axis!r}] = {out!r}")
    return out


def verify(axis: str) -> None:
    if axis not in HASHES:
        raise RuntimeError(f"NewBehav axis {axis!r} is not frozen yet; "
                           "collection or freeze() pending")
    for kind, want in HASHES[axis].items():
        p = DATA / f"newbehav_v1_{axis}_{kind}.json"
        got = _sha(p)
        if got != want:
            raise RuntimeError(f"{p.name}: hash {got} != frozen {want}")


# ---------------------------------------------------------------- validators
def validate_axis(axis: str, items: list[dict]) -> list[str]:
    """Structural + balance validation. Returns a list of violations."""
    errs: list[str] = []
    spec = AXES[axis]
    classes = spec["classes"]
    by_cls: dict[str, list[dict]] = {c: [] for c in classes}
    by_dom: dict[str, int] = {d: 0 for d in DOMAINS}
    seen_q: dict[str, set[str]] = {}
    for i, it in enumerate(items):
        for k in ("question", "response", "cls", "domain"):
            if k not in it:
                errs.append(f"[{i}] missing key {k}")
        c, d = it.get("cls"), it.get("domain")
        if c not in classes:
            errs.append(f"[{i}] unknown class {c!r}")
            continue
        if d not in DOMAINS:
            errs.append(f"[{i}] unknown domain {d!r}")
            continue
        by_cls[c].append(it)
        by_dom[d] += 1
        seen_q.setdefault(it["question"], set()).add(c)
    n_cls = {c: len(v) for c, v in by_cls.items()}
    if len(set(n_cls.values())) > 1:
        errs.append(f"class imbalance: {n_cls}")
    lo, hi = min(by_dom.values()), max(by_dom.values())
    if hi - lo > max(2, 0.05 * hi):
        errs.append(f"domain imbalance: {by_dom}")
    # matched-question pairing where the design demands it
    if axis in ("len", "schema", "persona"):
        want = len(classes)
        bad = sum(1 for q, cs in seen_q.items() if len(cs) != want)
        if bad:
            errs.append(f"{bad} questions not paired across all "
                        f"{want} classes (matched-question design)")
    return errs
