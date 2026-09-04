#!/usr/bin/env bash
# Bootstrap the GPU server (Ubuntu + NVIDIA driver already installed).
# Idempotent. Run from the repo root on the server:  bash server/setup.sh
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "== installing uv =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

echo "== venv + deps =="
uv venv .venv --python 3.12 -q --allow-existing
uv pip install -q --python .venv/bin/python -e '.[dev,gpu]'

echo "== sanity =="
.venv/bin/python - <<'PY'
import torch, bitsandbytes as bnb, transformers, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("vram", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
print("bnb", bnb.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
PY

echo "== unit tests =="
.venv/bin/python -m pytest -q

echo "setup OK"
