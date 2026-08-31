# In-Cell Learning

Code release for **"In-Cell Learning: Deployed Language Models Can Learn New
Knowledge Without Changing a Single Stored Bit"** (arXiv:2608.20873).

A quantized release defines, for every weight, a cell: the set of values
that round to the same stored code under the artifact's own scales. Serving
`w = anchor + M * tanh(s * z)` keeps every weight strictly inside its cell,
so the model learns while the shipped file's bytes — codes and scales —
remain bit-identical, verifiable by integer-domain re-quantization.

## Layout

- `cellfill/` — cell geometry per grid family (NF4, uniform W4A16, GGUF):
  walls, half-widths, invariance checks, packed serving, clip-merge, codec.
- `experiments/` — the paper's experiment drivers (single-shot, sequential
  writing with minor/major versions, fusion, capacity, controls).
- `scripts/` — static envelopes, witnesses, and table generation.

## Quick start

```bash
python experiments/exp0_clip_rate.py --model Qwen/Qwen3-1.7B-Base --n-facts 1000
```

Loads the published 4-bit release, trains a LoRA on synthetic facts,
clip-merges it into the frozen cells, and verifies invariance in the
integer domain.

## Citation

```bibtex
@article{incell2026,
  title={In-Cell Learning: Deployed Language Models Can Learn New Knowledge
         Without Changing a Single Stored Bit},
  journal={arXiv preprint arXiv:2608.20873},
  year={2026}
}
```

## License

Free for research and educational use; commercial use requires written
authorization — see [LICENSE](LICENSE).
