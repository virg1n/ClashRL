#!/usr/bin/env bash
# Builds everything the A/B pipeline needs, without sudo and without touching the root disk:
# a Python 3.12 env (uv), the packages in abtrain/requirements.txt (torch from PyPI), and a
# FirstLight_CR checkout pinned at 3ade158. Idempotent: a finished step is skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AB_WORK="${AB_WORK:-$HOME/clashrl_ab}"
source "$REPO/abtrain/env.sh"

VENV="$AB_WORK/venv"
FL="$AB_WORK/ext/FirstLight_CR"
FL_COMMIT=3ade158
LOG="$AB_WORK/logs/setup.log"
mkdir -p "$AB_WORK/logs"
say() { echo "$(date '+%F %T') setup: $*" | tee -a "$LOG"; }

retry() {  # retry <n> <command...>
  local n=$1; shift
  local i
  for i in $(seq 1 "$n"); do
    if "$@"; then return 0; fi
    say "attempt $i of $n failed: $*"; sleep $((i * 20))
  done
  return 1
}

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
  if [ -x "$AB_WORK/bin/uv" ]; then echo "$AB_WORK/bin/uv"; return; fi
  mkdir -p "$AB_WORK/bin"
  local url="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
  if curl -fsSL "$url" | tar -xz -C "$AB_WORK/bin" --strip-components=1 2>>"$LOG"; then
    echo "$AB_WORK/bin/uv"; return
  fi
  echo ""
}

env_ok() {
  [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - <<'EOF' >/dev/null 2>&1
import torch, numpy, pyarrow, huggingface_hub, royalesim, royalegym, royaleimitate, orjson, sc_compression
EOF
}

if env_ok; then
  say "python env ok ($VENV)"
else
  say "building the python env in $VENV"
  UV="$(find_uv)"
  # AB_TORCH_INDEX (optional, e.g. https://download.pytorch.org/whl/cpu for a CPU-only test box)
  # installs torch from that index first; by default torch comes from PyPI with CUDA bundled
  if [ -n "$UV" ]; then
    retry 3 "$UV" venv --seed --python 3.12 "$VENV" >>"$LOG" 2>&1
    if [ -n "${AB_TORCH_INDEX:-}" ]; then
      retry 3 "$UV" pip install --python "$VENV/bin/python" torch --index-url "$AB_TORCH_INDEX" >>"$LOG" 2>&1
    fi
    retry 3 "$UV" pip install --python "$VENV/bin/python" -r "$REPO/abtrain/requirements.txt" >>"$LOG" 2>&1
  else
    say "uv unavailable: falling back to the system python3 ($(python3 --version 2>&1))"
    python3 -m venv "$VENV" >>"$LOG" 2>&1
    retry 3 "$VENV/bin/python" -m pip install -q --upgrade pip >>"$LOG" 2>&1
    if [ -n "${AB_TORCH_INDEX:-}" ]; then
      retry 3 "$VENV/bin/python" -m pip install -q torch --index-url "$AB_TORCH_INDEX" >>"$LOG" 2>&1
    fi
    retry 3 "$VENV/bin/python" -m pip install -q -r "$REPO/abtrain/requirements.txt" >>"$LOG" 2>&1
  fi
  env_ok || { say "python env still broken, see $LOG"; exit 1; }
  say "python env ready"
fi

# FirstLight (only for the stand-ins and the evaluation vs FL General). A failure here is not
# fatal: the pipeline retries the clone itself (stage fl_clone) and trains meanwhile.
bash "$REPO/abtrain/fl_clone.sh" "$FL" "$FL_COMMIT" >>"$LOG" 2>&1 || say "WARNING: FirstLight clone failed; the pipeline will retry it"
FL_OK=0
[ -s "$FL/checkpoints/General/checkpoint-step-00000460.pt" ] && FL_OK=1

"$VENV/bin/python" - <<EOF 2>&1 | tee -a "$LOG"
import sys, torch
sys.path[:0] = ["$FL", "$REPO/firstlight-port", "$REPO"]
if $FL_OK:
    import flport.agent  # noqa: F401  (FirstLight + flport import cleanly)
from abtrain.game import Catalog, make_engine
from abtrain.model import build_model, param_count
cat = Catalog.build(make_engine())
print(f"setup check: torch {torch.__version__}, {torch.cuda.device_count()} GPUs, {len(cat.names)} cards, "
      f"model {param_count(build_model()) / 1e6:.2f}M params")
EOF
say "setup complete"
