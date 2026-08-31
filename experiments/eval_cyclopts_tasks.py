#!/usr/bin/env python
"""What does a probe point buy in task points?

Every gain this project reports on injected knowledge is a probe metric: the
model ranks the right signature above distractors. The quantity nobody has
measured is the exchange rate into work -- whether a model that scores twenty
points higher on cyclopts probes writes cyclopts programs that run.

The measurement has to be made where the injected knowledge is actually
required, or the denominator is zero. Terminal-Bench is the wrong place for
it: its tool surfaces come back already held by the scan, so a knowledge fill
has nothing to supply and a null there confirms the scan rather than pricing
the exchange. cyclopts is the right place: the 27B release holds 12.1% of it,
the fill takes that to 31.2%, and a program that uses cyclopts wrongly does
not run.

Each task is a short specification, an argv, and the exact stdout a correct
program produces. The model writes the program, the program is executed, and
the task passes only if it runs and prints what was asked. Nothing is scored
by inspecting the source, because a call that looks right and raises is a
failure the probe metric would have counted as a success.

  python experiments/eval_cyclopts_tasks.py --model <release> --load-4bit
  python experiments/eval_cyclopts_tasks.py --model <release> --fill out/fill_27b_cyclopts.pt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Each task names a cyclopts feature the corpus teaches. The reference shape
# was executed before the set was written, so a passing program is known to
# exist for every one of them.
TASKS = [
    # --- calibration note ---------------------------------------------------
    # The first version of this set asked for ordinary CLI programs and the
    # released 27B passed every valid one of them. That is a real finding --
    # a model holding 12.1% of cyclopts by probe still writes cyclopts
    # programs that run, so recalling a signature and using a library are not
    # the same ability -- but it leaves no headroom in which to price the
    # exchange from probe points to task points.
    #
    # Every task below therefore turns on a cyclopts-SPECIFIC parameter name,
    # one that general CLI convention does not supply and that a model has to
    # have learned: Parameter's alias/negative/converter/validator/name, and
    # App's version/default_parameter. Each reference shape was executed
    # before the set was written.
    dict(id="param-alias",
         spec="Write a Python CLI using cyclopts. The default command takes "
              "an integer option `value` (default 0) that must ALSO be "
              "reachable as the short flag `-v`. Print value times 10. Use "
              "typing.Annotated with cyclopts.Parameter to attach the extra "
              "name.",
         argv=["-v", "7"], expect="70"),
    dict(id="param-negative",
         spec="Write a Python CLI using cyclopts. The default command takes a "
              "keyword-only boolean `debug` defaulting to True, whose "
              "negative form must be spelled `--nodebug` rather than the "
              "default `--no-debug`. Print `on` when debug is true and `off` "
              "otherwise. Use Annotated with cyclopts.Parameter.",
         argv=["--nodebug"], expect="off"),
    dict(id="param-converter",
         spec="Write a Python CLI using cyclopts whose default command takes "
              "an integer `n` (default 0) whose value is doubled during "
              "parsing by a custom conversion function attached through "
              "cyclopts.Parameter. The function receives the annotated type "
              "and the list of tokens. Print n.",
         argv=["21"], expect="42"),
    dict(id="param-validator",
         spec="Write a Python CLI using cyclopts whose default command takes "
              "an integer `n` (default 1) with a validation function attached "
              "through cyclopts.Parameter that raises ValueError when the "
              "value is not positive. The function receives the annotated "
              "type and the converted value. Print n.",
         argv=["5"], expect="5"),
    dict(id="param-rename",
         spec="Write a Python CLI using cyclopts whose default command has a "
              "parameter called `the_value` (integer, default 0) that must be "
              "given on the command line as `--amount`. Print the value plus "
              "one. Use Annotated with cyclopts.Parameter to set the name.",
         argv=["--amount", "41"], expect="42"),
    dict(id="app-version",
         spec="Write a Python CLI using cyclopts. Construct the App so that "
              "it reports the version string `9.9.9` when asked, and give it "
              "a default command that prints `ran`.",
         argv=["--version"], expect="9.9.9"),
    dict(id="app-default-parameter",
         spec="Write a Python CLI using cyclopts. Configure the App so that "
              "boolean options get NO auto-generated negative form at all, by "
              "passing a default parameter configuration at App construction. "
              "The default command takes a keyword-only boolean `flag` "
              "defaulting to False and prints `yes` when set, `no` otherwise.",
         argv=["--flag"], expect="yes"),
    # a small number of ordinary tasks are kept as a floor: if a model fails
    # these it is failing at Python rather than at cyclopts, and the harder
    # rows above cannot be read.
    dict(id="floor-default",
         spec="Write a Python CLI using the cyclopts library. Create an App "
              "and register a default command that takes a positional "
              "argument `name` (a string) and prints exactly `hello <name>`.",
         argv=["world"], expect="hello world"),
    dict(id="floor-commands",
         spec="Write a Python CLI using the cyclopts library that registers "
              "two named commands with @app.command: `add` which takes two "
              "integers a and b and prints their sum, and `mul` which takes "
              "two integers a and b and prints their product.",
         argv=["mul", "6", "7"], expect="42"),
]

PROMPT = (
    "Write a complete Python program that satisfies the specification.\n"
    "Use the `cyclopts` library. Output only code, no explanation.\n"
    "The program must call the app when run as a script.\n\n"
    "Specification: {spec}\n\n"
    "```python\n"
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", default=None)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--samples", type=int, default=3,
                   help="generations per task; a task passes if any sample "
                        "runs correctly, which is pass@k and is the metric a "
                        "developer actually experiences")
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default="out/cyclopts_tasks.json")
    return p.parse_args()


def extract_code(text: str) -> str:
    """The first fenced block, or everything up to the first fence close."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text.split("```")[0]


def run_program(code: str, argv, python, timeout):
    """Execute the program and return (ok, stdout, note)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(code)
        try:
            r = subprocess.run([python, str(f), *argv], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
        if r.returncode != 0:
            return False, r.stdout, f"exit {r.returncode}: {r.stderr.strip()[-120:]}"
        return True, r.stdout, ""


def main():
    a = parse()
    t0 = time.time()
    import torch

    from experiments.served import load_served

    if a.load_4bit and not a.fill:
        from experiments.exp0_clip_rate import build_4bit
        model, tok = build_4bit(a.model)
        model.eval()
        label = "released-4bit"
    else:
        model, tok, label = load_served(a.model, None,
                                        dtype=getattr(torch, a.dtype),
                                        fill=a.fill)
    print(f"[arm] {label}", flush=True)

    from experiments.served import generate_batch

    results, passed = [], 0
    for t in TASKS:
        prompt = PROMPT.format(spec=t["spec"])
        outs = generate_batch(model, tok, [prompt] * a.samples,
                              max_new=a.max_new, bs=a.samples)
        ok_any, notes = False, []
        for o in outs:
            code = extract_code(o)
            ok, stdout, note = run_program(code, t["argv"], a.python, a.timeout)
            if ok and stdout.strip() == t["expect"].strip():
                ok_any = True
                notes.append("pass")
                break
            notes.append(note or f"wrong output {stdout.strip()[:60]!r}")
        passed += ok_any
        results.append(dict(id=t["id"], passed=ok_any, notes=notes))
        print(f"  {t['id']:20s} {'PASS' if ok_any else 'fail':4s}  "
              f"{notes[-1][:70]}", flush=True)

    out = dict(model=a.model, fill=a.fill, arm=label,
               samples=a.samples, n_tasks=len(TASKS), passed=passed,
               pass_rate=passed / len(TASKS), results=results,
               minutes=(time.time() - t0) / 60)
    from experiments.exp0_clip_rate import stamp_of
    out["scorer"] = stamp_of(extract_code, run_program)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {label}: {passed}/{len(TASKS)} = {out['pass_rate']:.1%} "
          f"({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
