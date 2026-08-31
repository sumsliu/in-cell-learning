#!/usr/bin/env python
"""Coordinator of a federation of writers (experiments/fed_node.py).

Moves fills between machines and nothing else. Each node trains its own
shard of facts into the released model; the coordinator gathers the
round's fills from every node and scatters them to every node; each node
folds the merge itself and reports a hash of its anchors, which must
agree across the federation. Data never moves.

  python scripts/fed_run.py --run clinic --hosts hlink@192.168.50.201,... \
      --model unsloth/Qwen3-8B-Base-bnb-4bit --facts-file data/clinic_train.json \
      --probes-file data/clinic_probes.json --rounds 6 --epochs 24
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new"]


def sh(host, cmd, timeout=120):
    r = subprocess.run(SSH + [host, cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip()


def rsync(src, dst, timeout=900):
    subprocess.run(["rsync", "-az", "-e", " ".join(SSH), src, dst], check=True, timeout=timeout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--hosts", required=True, help="comma-separated user@host, one per node")
    p.add_argument("--model", required=True)
    p.add_argument("--facts-file", required=True)
    p.add_argument("--probes-file", required=True)
    p.add_argument("--shards", default=None)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--merge", default="average")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--extra", default="", help="extra fed_node flags")
    p.add_argument("--remote-dir", default="~/clora")
    p.add_argument("--out", default=None)
    p.add_argument("--no-launch", action="store_true",
                   help="the nodes are started by their own queues; only move fills")
    args = p.parse_args()
    hosts = args.hosts.split(",")
    K = len(hosts)
    local = Path("fed") / args.run
    local.mkdir(parents=True, exist_ok=True)
    # data to every node, then launch
    for k, h in enumerate(hosts):
        rsync(args.facts_file, f"{h}:{args.remote_dir}/data/")
        rsync(args.probes_file, f"{h}:{args.remote_dir}/data/")
        if args.no_launch:
            continue
        shards = f"--shards {args.shards}" if args.shards else ""
        cmd = (f"cd {args.remote_dir} && source lane_common.sh && rm -rf fed/{args.run} && "
               f"setsid -f $PY experiments/fed_node.py --run {args.run} --node {k} --nodes {K} "
               f"--model {args.model} --facts-file {args.facts_file} --probes-file {args.probes_file} "
               f"{shards} --rounds {args.rounds} --epochs {args.epochs} --merge {args.merge} --bs {args.bs} "
               f"{args.extra} > out/fed_{args.run}_node{k}.log 2>&1 < /dev/null")
        # the remote process must outlive the ssh session: run it detached
        # and do not wait for the session's pipes to close
        subprocess.Popen(SSH + [h, cmd], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        rc, out = sh(h, f"pgrep -f 'fed_node.py --run {args.run} ' | wc -l")
        print(f"[launch] node {k} on {h}: {out.strip()} process(es)", flush=True)
    for rnd in range(args.rounds):
        rdir = local / f"round_{rnd}"
        rdir.mkdir(exist_ok=True)
        # every node's fill must exist where it was written
        for k, h in enumerate(hosts):
            remote = f"{args.remote_dir}/fed/{args.run}/round_{rnd}/fill_{k}.pt"
            while True:
                rc, out = sh(h, f"test -f {remote} && echo yes || (tail -n 1 {args.remote_dir}/out/fed_{args.run}_node{k}.log 2>/dev/null | cut -c1-80)")
                if out.strip() == "yes":
                    break
                if "Traceback" in out or "Error" in out:
                    raise SystemExit(f"node {k} failed: {out}")
                time.sleep(30)
            print(f"[round {rnd}] fill {k} in place on {h}", flush=True)
        # scatter peer-to-peer: each source pushes its own fill to the other
        # nodes directly (the machines share a switch; relaying every file
        # through the coordinator made the round 30 minutes longer)
        pushes = []
        for k, h in enumerate(hosts):
            dsts = " ".join(hosts[j].split("@")[-1] if "@" not in hosts[j] else hosts[j]
                            for j in range(K) if j != k)
            cmd = (f"for d in {dsts}; do rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' "
                   f"{args.remote_dir}/fed/{args.run}/round_{rnd}/fill_{k}.pt "
                   f"$d:{args.remote_dir}/fed/{args.run}/round_{rnd}/ & done; wait; echo pushed")
            pushes.append(subprocess.Popen(SSH + [h, cmd], stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for pr in pushes:
            pr.wait()
        print(f"[round {rnd}] peer-scattered {K} fills", flush=True)
        # archive to the coordinator for the replay audit (off the hot path
        # only in the sense that the nodes are already folding)
        for k, h in enumerate(hosts):
            if not (rdir / f"fill_{k}.pt").exists():
                rsync(f"{h}:{args.remote_dir}/fed/{args.run}/round_{rnd}/fill_{k}.pt",
                      str(rdir / f"fill_{k}.pt"))
        print(f"[round {rnd}] archived {K} fills", flush=True)
        # agreement
        hashes = {}
        for k, h in enumerate(hosts):
            remote = f"{args.remote_dir}/fed/{args.run}/round_{rnd}/state_{k}.txt"
            while True:
                rc, out = sh(h, f"cat {remote} 2>/dev/null")
                if out:
                    hashes[k] = out.strip()
                    break
                time.sleep(20)
        agree = len(set(hashes.values())) == 1
        print(f"[round {rnd}] anchors identical on all nodes: {agree} ({hashes[0][:16]})", flush=True)
    finals = {}
    for k, h in enumerate(hosts):
        remote = f"{args.remote_dir}/fed/{args.run}/final_{k}.json"
        while sh(h, f"test -f {remote} && echo yes")[1] != "yes":
            time.sleep(20)
        rsync(f"{h}:{remote}", str(local / f"final_{k}.json"))
        finals[k] = json.loads((local / f"final_{k}.json").read_text())
    summary = dict(run=args.run, hosts=hosts, model=args.model, rounds=args.rounds,
                   shards={k: f["shard"] for k, f in finals.items()},
                   pooled_recall={k: f["final"]["pooled"] for k, f in finals.items()},
                   by_shard=finals[0]["final"]["by_shard"],
                   anchors_sha256={k: f["final"]["anchors_sha256"] for k, f in finals.items()},
                   identical=len({f["final"]["anchors_sha256"] for f in finals.values()}) == 1,
                   violations={k: f["invariance_violations"] for k, f in finals.items()},
                   ppl={k: f["ppl"] for k, f in finals.items()},
                   ppl_lambada={k: f["ppl_lambada"] for k, f in finals.items()},
                   history=finals[0]["history"], minutes={k: f["minutes"] for k, f in finals.items()})
    out = Path(args.out or f"results/fed_{args.run}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k in ("pooled_recall", "identical", "violations", "by_shard")}, indent=1))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
