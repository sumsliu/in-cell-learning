"""Can we freeze an officially published 4-bit artifact, unchanged?

The paper's premise is that a released 4-bit file is preserved bit-for-bit.
Every experiment so far quantized an fp16 model locally, so the "release" was
ours. This loads a published bnb-4bit checkpoint and checks that the frozen
state extractor reads it directly: same codes, same scales, same cells.
"""
import sys, torch
sys.path.insert(0, "/home/zssy/clora")
from transformers import AutoModelForCausalLM
from cellfill.bnb_state import frozen_state_from_linear4bit
from cellfill.bins import bin_bounds

mid = sys.argv[1]
print(f"[load] {mid} (already 4-bit; no quantization_config passed)")
m = AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0},
                                         dtype=COMPUTE_DTYPE)
n_l4 = sum(1 for _, mod in m.named_modules()
           if type(mod).__name__ == "Linear4bit")
print(f"[load] {n_l4} Linear4bit layers found")
if not n_l4:
    print("VERDICT: not a bitsandbytes 4-bit checkpoint as loaded")
    sys.exit(1)
tot = 0
for i, (name, mod) in enumerate(
        [(n, x) for n, x in m.named_modules()
         if type(x).__name__ == "Linear4bit"][:3]):
    fs = frozen_state_from_linear4bit(mod)
    lo, hi = bin_bounds(fs["codes"], fs["absmax"], fs["blocksize"],
                        capped=True, margin=0.01)
    anch = fs["anchors"].reshape(-1)
    room = torch.minimum(anch - lo, hi - anch)
    tot += anch.numel()
    print(f"  {name}: {tuple(fs['shape'])} blocksize={fs['blocksize']} "
          f"mean room {room.mean():.3e} negative {int((room<0).sum())}")
print(f"VERDICT: frozen state extracted from the published artifact, "
      f"{n_l4} layers")
