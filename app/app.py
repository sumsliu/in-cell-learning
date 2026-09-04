#!/usr/bin/env python
"""CellFill: drop knowledge in, press run, use the model.

A single page in three parts. INJECT: drop files, see the sentences the
platform will write, optionally have the model restate each one, press
Run; the training job runs on a second GPU with its log streaming, and
ends with a card: recall on the sentences' own completions, drift on
WikiText and LAMBADA, the fill's size, and the invariance check on every
weight. CHAT: the same question answered by the released model, the
injected model, retrieval over the dropped sentences, or both, side by
side. VERIFY: the served file's hash, the in-cell check, and the rollback
button, which is a subtraction.

  python app/app.py --model /home/zssy/models/Qwen3.8-27B --serve-gpu 0 --train-gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from app.corpus import (  # noqa: E402
    digest, probes_of, read_file, sentences_of, write_corpus,
)
from app.engine import Engine  # noqa: E402

ARGS = None
ENG: Engine | None = None
STATE = dict(sents=[], probes=[], train_json=None, probes_json=None,
             fill=None, report=None, base_recall=None)


def on_files(files):
    if not files:
        return "", gr.update(value="")
    sents = []
    for f in files:
        sents += sentences_of(read_file(f.name if hasattr(f, "name") else f))
    STATE["sents"] = sents
    STATE["probes"] = probes_of(sents, ENG.tok)
    preview = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents[:40]))
    more = f"\n... ({len(sents)} sentences in all)" if len(sents) > 40 else ""
    return (f"{len(sents)} sentences, {len(STATE['probes'])} completion "
            f"probes from {len(files)} file(s)"), preview + more


def on_augment(n):
    sents = STATE["sents"]
    if not sents:
        return "drop files first", ""
    extra = ENG.paraphrase(sents, n=int(n))
    STATE["sents"] = sents + extra
    STATE["probes"] = probes_of(sents, ENG.tok)   # probes stay on the originals
    return (f"{len(sents)} sentences + {len(extra)} restatements by the "
            f"released model"), "\n".join(STATE["sents"][:60])


def run_training(epochs, rank, lr, progress=gr.Progress()):
    sents, probes = STATE["sents"], STATE["probes"]
    if not sents:
        yield "drop files first", ""
        return
    name = f"kb_{int(time.time())}"
    out_dir = ROOT / "app" / "runs" / name
    train_json, probes_json = write_corpus(sents, probes, str(out_dir), name)
    fill = out_dir / "fill.pt"
    log_path = out_dir / "train.log"
    base_recall = ENG.recall(probes, use_fill=False)
    STATE["base_recall"] = base_recall
    cmd = [sys.executable, str(ROOT / "experiments" / "exp5_qil.py"),
           "--model", ARGS.model, "--facts-file", train_json,
           "--probes-file", probes_json, "--epochs", str(int(epochs)),
           "--rank", str(int(rank)), "--tanh-scale", "40", "--lr", str(lr),
           "--replay-frac", "0.1", "--seed", "0", "--bs", str(ARGS.train_bs),
           "--codebook-m", "--checkpoint-fill", "--map-dtype", "float16",
           "--eval-dtype", "bfloat16", "--skip-anchor-eval", "--inplace-only",
           "--max-ppl-chunks", "20", "--save-fill", str(fill),
           "--out", str(out_dir / "result.json")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ARGS.train_gpu),
               HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, cwd=str(ROOT))
    status = f"training {len(sents)} sentences on GPU {ARGS.train_gpu} ..."
    while proc.poll() is None:
        time.sleep(5)
        tail = "".join(Path(log_path).read_text(errors="ignore").splitlines(True)[-25:])
        yield status, tail
    tail = "".join(Path(log_path).read_text(errors="ignore").splitlines(True)[-40:])
    if proc.returncode != 0 or not fill.exists():
        yield f"training failed (exit {proc.returncode}); see the log", tail
        return
    n = ENG.attach(str(fill))
    res = json.loads((out_dir / "result.json").read_text())
    recall = ENG.recall(probes, use_fill=True)
    n_bad, n_tot = ENG.verify()
    ENG.index(sents)
    rep = dict(name=name, sentences=len(sents), probes=len(probes),
               recall_before=base_recall, recall_after=recall,
               wikitext=res["ppl"].get("merged") or res["ppl"].get("trained_inplace"),
               lambada=res["ppl_lambada"].get("merged") or res["ppl_lambada"].get("trained_inplace"),
               fill_mb=round(fill.stat().st_size / 1e6, 1),
               invariance_violations=n_bad, weights_checked=n_tot,
               minutes=res.get("minutes"), layers=n)
    STATE.update(fill=str(fill), report=rep)
    card = (f"**{rep['sentences']} sentences written in {rep['minutes']} min.**  \n"
            f"completion recall: {100 * base_recall:.1f}% → "
            f"**{100 * recall:.1f}%**  \n"
            f"drift: WikiText {rep['wikitext']:.2f}, LAMBADA {rep['lambada']:.1f}  \n"
            f"fill: {rep['fill_mb']} MB over {n} layers  \n"
            f"invariance: **{n_bad} violations** in {n_tot:,} weights; the "
            f"served file is unchanged (sha256 {ARGS.file_hash[:16]}…)")
    yield card, tail


def on_chat(question, arms, k):
    if not question.strip():
        return "", "", "", ""
    outs = {}
    for arm in ("released", "injected", "retrieval", "injected + retrieval"):
        if arm not in arms:
            outs[arm] = ""
            continue
        use_fill = arm.startswith("injected") and STATE["fill"] is not None
        use_rag = "retrieval" in arm
        text, ctx = ENG.answer(question, use_fill=use_fill, use_rag=use_rag,
                               k=int(k))
        if use_rag and ctx:
            text += "\n\n[retrieved]\n" + "\n".join(f"- {c}" for c in ctx)
        outs[arm] = text
    return (outs["released"], outs["injected"], outs["retrieval"],
            outs["injected + retrieval"])


def on_verify():
    n_bad, n_tot = ENG.verify()
    return (f"served release: `{ARGS.model}`  \nsha256 of the file(s): "
            f"`{ARGS.file_hash}`  \nattached fill: `{STATE['fill'] or 'none'}`  \n"
            f"in-cell check: **{n_bad} violations** in {n_tot:,} weights")


def on_rollback():
    ENG.detach()
    STATE["fill"] = None
    return "fill detached: the served model is the release again (a subtraction)"


def main():
    global ARGS, ENG
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--serve-gpu", type=int, default=0)
    p.add_argument("--train-gpu", type=int, default=1)
    p.add_argument("--train-bs", type=int, default=4)
    p.add_argument("--no-codebook", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    ARGS = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.serve_gpu)
    # the hash of the served weights file(s): what "the same model" means
    mp = Path(ARGS.model)
    files = sorted(mp.glob("*.safetensors")) if mp.is_dir() else []
    ARGS.file_hash = (digest(str(files[0])) if files else "(hub release)")
    ENG = Engine(ARGS.model, codebook_m=not ARGS.no_codebook)

    with gr.Blocks(title="CellFill") as demo:
        gr.Markdown("# CellFill — write knowledge into a released 4-bit model\n"
                    f"Serving `{ARGS.model}` on GPU {ARGS.serve_gpu}; "
                    f"training on GPU {ARGS.train_gpu}. The served file never "
                    "changes; the knowledge lives in the rounding cells.")
        with gr.Tab("Inject"):
            files = gr.File(file_count="multiple", label="drop documents (txt, md, pdf)")
            info = gr.Markdown()
            preview = gr.Textbox(label="sentences to write", lines=12)
            with gr.Row():
                n_aug = gr.Slider(0, 4, value=2, step=1, label="restatements per sentence")
                aug_btn = gr.Button("restate with the model")
            with gr.Row():
                epochs = gr.Slider(4, 32, value=16, step=1, label="epochs")
                rank = gr.Dropdown([16, 32, 64], value=32, label="rank")
                lr = gr.Dropdown([1e-3, 5e-4], value=1e-3, label="learning rate")
            run_btn = gr.Button("Run", variant="primary")
            card = gr.Markdown()
            log = gr.Textbox(label="training log", lines=14)
            files.change(on_files, files, [info, preview])
            aug_btn.click(on_augment, n_aug, [info, preview])
            run_btn.click(run_training, [epochs, rank, lr], [card, log])
        with gr.Tab("Chat"):
            q = gr.Textbox(label="ask", lines=2)
            arms = gr.CheckboxGroup(["released", "injected", "retrieval",
                                     "injected + retrieval"],
                                    value=["released", "injected", "retrieval"],
                                    label="answer with")
            k = gr.Slider(1, 5, value=3, step=1, label="retrieved sentences")
            ask = gr.Button("ask", variant="primary")
            with gr.Row():
                a0 = gr.Textbox(label="released", lines=6)
                a1 = gr.Textbox(label="injected", lines=6)
            with gr.Row():
                a2 = gr.Textbox(label="retrieval", lines=6)
                a3 = gr.Textbox(label="injected + retrieval", lines=6)
            ask.click(on_chat, [q, arms, k], [a0, a1, a2, a3])
        with gr.Tab("Verify"):
            vbtn = gr.Button("check the served model")
            vout = gr.Markdown()
            rbtn = gr.Button("roll back (detach the fill)")
            rout = gr.Markdown()
            vbtn.click(on_verify, None, vout)
            rbtn.click(on_rollback, None, rout)
    demo.queue().launch(server_name="0.0.0.0", server_port=ARGS.port, share=False)


if __name__ == "__main__":
    main()
