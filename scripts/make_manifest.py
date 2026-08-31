#!/usr/bin/env python
"""Generate the run manifest: every archived result file, what it was, and
where its numbers appear. A reader should be able to take any number in the
paper and find the file it came from without asking us."""

import json
from pathlib import Path

R = Path("results")

# run stem -> (what it measures, which section uses it)
WHERE = {
    "exp0_qwen3-1.7b_r16_e8": ("clip rate at 8 epochs", "5.4"),
    "exp0b_qwen3-1.7b_r16_e24": ("no rehearsal, 24 epochs", "5.2, 5.5"),
    "exp0c_qwen3-1.7b_r16_e24_origbase": ("merge base = original fp32", "5.4"),
    "exp1_replay30_qwen3-1.7b_r16_e24": ("stale rehearsal buffer", "5.3, Fig 1"),
    "exp1b_freshreplay30_qwen3-1.7b_r16_e24": ("fresh rehearsal 30%", "5.3"),
    "exp1c_replay10_qwen3-1.7b_r16_e24": ("A / A+ seed 0", "Table 1"),
    "exp1c_s1": ("A / A+ seed 1", "Table 1, 5.6"),
    "exp1c_s2": ("A / A+ seed 2", "Table 1, 5.6"),
    "exp2_heal4_qwen3-1.7b_r16_e24": ("A+ at rehearsal 30%", "5.7"),
    "exp3_dense_qwen3-1.7b_e24": ("B dense seed 0", "Table 1"),
    "exp3_s1": ("B dense seed 1", "Table 1"),
    "exp3_s2": ("B dense seed 2", "Table 1"),
    "exp4_free_qwen3-1.7b_e24": ("unconstrained seed 0", "Table 1, 5.5"),
    "exp4_free_s1": ("unconstrained seed 1", "Table 1, 5.5"),
    "exp4_free_s2": ("unconstrained seed 2", "Table 1, 5.5"),
    "exp5_qil_s4_qwen3-1.7b_r16_e24": ("CellFill, under-stepped gain", "5.4"),
    "exp5b_qil_s40_qwen3-1.7b_r16_e24": ("CellFill r16 at rehearsal 30%", "5.7"),
    "exp5b_qil_s1": ("CellFill r16 seed 1", "5.7"),
    "exp5b_qil_s2": ("CellFill r16 seed 2", "5.7"),
    "exp6_10k_qwen3-1.7b_r16_e24": ("capacity 10k, A and A+", "5.8"),
    "exp6b_dense10k_qwen3-1.7b_e24": ("capacity 10k, B", "5.8, Fig 2"),
    "exp7_rho05_qwen3-1.7b_r16_e24": ("radius rho = 0.5", "5.4"),
    "exp8_e96_qwen3-1.7b": ("exposure 96 epochs", "5.7, 5.6"),
    "exp10_qil_r64_qwen3-1.7b": ("CellFill r64", "Table 1, 5.7"),
    "exp16_qil_r16_replay10": ("CellFill r16, matched rehearsal", "Table 1, 5.7"),
    "exp12_27b_e8": ("27B, 8 epochs", "5.11, 5.6"),
    "exp12b_27b_e24": ("27B, 24 epochs", "5.11"),
    "exp14_gateup": ("partition: gate+up", "5.10"),
    "exp14_mlp": ("partition: all MLP", "5.10"),
    "exp14_attn": ("partition: attention", "5.10"),
    "exp14_down": ("partition: down_proj", "5.10"),
    "exp14_early": ("partition: layers 0-8", "5.10"),
    "exp14_mid": ("partition: layers 9-18", "5.10"),
    "exp14_late": ("partition: layers 19-27", "5.10"),
    "exp15_seq_minor": ("4 sequential tasks", "5.12"),
    "exp15_seq_consolidate": ("4 tasks with consolidation", "5.12"),
    "exp17_b10k_xdom": ("capacity 10k, B, cross-domain", "5.8, 6"),
    "exp17_aplus10k_xdom": ("capacity 10k, A+ r64, cross-domain", "5.8"),
    "exp_geom_1p7b": ("safe-radius geometry, 1.7B", "5.9, Fig 3"),
    "exp_geom_4b": ("safe-radius geometry, 4B", "5.9"),
    "exp18_b_replay10_s0": ("B dense at rehearsal 0.1, seed 0", "Table 1"),
    "exp18_b_replay10_s1": ("B dense at rehearsal 0.1, seed 1", "Table 1"),
    "exp18_b_replay10_s2": ("B dense at rehearsal 0.1, seed 2", "Table 1"),
    "exp20_qil_0p6": ("0.6B CellFill r64", "5.11"),
    "exp20_aplus_0p6": ("0.6B A+ heal", "5.6"),
    "exp21_qil_4b": ("4B CellFill r64", "5.11"),
    "exp23_mistral_a": ("Mistral-7B clip-merge, diverged", "5.11"),
    "exp23_mistral_qil": ("Mistral-7B CellFill r64", "5.11"),
    "exp24_qil_8b": ("8B CellFill r64", "5.11"),
    "exp25_mistral_a_lr5e-5": ("Mistral-7B clip-merge at lr 5e-5", "5.11"),
    "exp26_lora_lr1e-3": ("stability sweep: LoRA at lr 1e-3", "5.11"),
    "exp26_lora_lr3e-3": ("stability sweep: LoRA at lr 3e-3, diverged", "5.11"),
    "exp26_cellfill_lr1e-3": ("stability sweep: CellFill at lr 1e-3", "5.11"),
    "exp26_cellfill_lr3e-3": ("stability sweep: CellFill at lr 3e-3", "5.11"),
    "exp29_seq_etanh": ("4 tasks, E|tanh| logged at fold", "5.12"),
    "exp30_qil_r64_s1": ("CellFill r64 seed 1", "Table 1, 5.7"),
    "exp31_qil_r16_s1": ("CellFill r16 seed 1, rehearsal 0.1", "5.7"),
    "exp31_qil_r16_s2": ("CellFill r16 seed 2, rehearsal 0.1", "5.7"),
    "exp33_real_cellfill": ("real corpus, CellFill r64, 1 phrasing", "5.4"),
    "exp34_real_aplus": ("real corpus, clip-merge then heal", "5.4"),
    "exp35_real_free": ("real corpus, unconstrained control", "5.4"),
    "exp36_realaug_cellfill": ("real corpus, medical domain augmented", "5.4"),
    "exp37_realaug_all": ("real corpus, all domains augmented", "5.4"),
    "exp38_real8dom": ("8 domains, truncated generation budget", "5.4"),
    "exp40_real8_fixed": ("8 domains, budget from the data", "5.4"),
    "exp41_real8_final": ("8 domains, per-domain floors archived", "5.4"),
    "bench_released": ("ARC/HellaSwag/WinoGrande, fp32 release", "5.5"),
    "bench_anchor": ("ARC/HellaSwag/WinoGrande, 4-bit anchor", "5.5"),
    "bench_merged": ("ARC/HellaSwag/WinoGrande, served model", "5.5"),
    "exp30_qil_r64_s2": ("CellFill r64 seed 2", "Table 1, 5.7"),
    "exp_codec": ("nested checkpoints, k = 1..4", "6"),
    "exp119_loop_rehearse": ("the full cycle as a loop: six tasks, rehearsal, "
                             "consolidating every second task", "5.5"),
    "exp119_loop_plain": ("the same loop with rehearsal off: what consolidation "
                          "buys and what it does not", "5.5"),
    "exp70_seq_preimage": ("six tasks in a fixed cell: the room never decays",
                           "5.5"),
    "exp70_seq_anchor": ("six tasks folded, the control the fixed cell is read "
                         "against", "5.5"),
    "exp122_validate_s10": ("the gentle recipe with the task suite scored inside "
                            "the sequence and a recovery pass after each major",
                            "5.5"),
    "exp122_validate_s40": ("the same, at the standard recipe: where the trade "
                            "between capability and knowledge lands", "5.5"),
    "exp121_s10_loop_rehearse": ("the gentle recipe through the same cycle: "
                                 "where consolidation stops paying", "5.5"),
    "exp120_diag_minor": ("six folds with the displacement's distribution kept, "
                          "not just its mean: what binds a long sequence", "5.5"),
    "curvature_cells_1p7b": ("do the cells sit where the loss is curved? "
                             "Fisher against room, per weight", "5.1"),
}

rows = []
# results/ holds every run this repository has ever archived, including work
# that belongs to other manuscripts. The manifest lists only the runs WHERE
# names, so regenerating it can never widen what the paper discloses.
skipped, unlisted = [], []
for p in sorted(R.glob("*.json")):
    stem = p.stem
    if stem not in WHERE:
        unlisted.append(stem)
        continue
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        # a truncated or empty archive is a defect to surface, not to embed
        skipped.append(stem)
        continue
    cfg = d.get("config") or {}
    model = str(cfg.get("model", "")).split("/")[-1] or "---"
    what, _sec = WHERE.get(stem, ("(unmapped)", "---"))
    mins = d.get("minutes")
    if not isinstance(mins, (int, float)):
        mins = None
    esc = lambda t: (t.replace("\\", "").replace("_", "\\_")
                     .replace("%", "\\%").replace("&", "\\&"))
    rows.append((f"\\texttt{{{esc(stem)}}}", esc(model), esc(what),
                 f"{mins:.0f}" if mins else "---"))

out = ["% GENERATED by scripts/make_manifest.py -- do not edit by hand",
       "\\begin{tabular}{llp{0.34\\linewidth}r}", "\\toprule",
       "archived file & model & what it measures & min \\\\",
       "\\midrule"]
out += [" & ".join(r) + " \\\\" for r in rows]
out += ["\\bottomrule", "\\end{tabular}"]
Path("paper/tables/manifest.tex").write_text("\n".join(out) + "\n")
print(f"wrote paper/tables/manifest.tex with {len(rows)} runs")
unmapped = [r[0] for r in rows if "unmapped" in r[2]]
print("unmapped:", unmapped or "none")
print("unreadable, skipped:", skipped or "none")
print(f"archived but not listed in this paper: {len(unlisted)}")
