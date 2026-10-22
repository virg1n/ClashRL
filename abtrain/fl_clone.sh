#!/usr/bin/env bash
# fl_clone.sh <dir> <commit>: FirstLight_CR checked out at <commit> (its checkpoints are plain git files).
set -euo pipefail
FL="$1"; COMMIT="$2"
for i in 1 2 3; do
  if [ ! -d "$FL/.git" ]; then
    rm -rf "$FL"; mkdir -p "$(dirname "$FL")"
    git clone -q https://github.com/jaasssoooonnnnn/FirstLight_CR "$FL" || { sleep $((i * 30)); continue; }
  fi
  if [ "$(git -C "$FL" rev-parse --short=7 HEAD)" != "$COMMIT" ]; then
    git -C "$FL" checkout -q "$COMMIT" 2>/dev/null || { git -C "$FL" fetch -q origin && git -C "$FL" checkout -q "$COMMIT"; } || { sleep $((i * 30)); continue; }
  fi
  [ -s "$FL/checkpoints/General/checkpoint-step-00000460.pt" ] && [ -s "$FL/checkpoints/2_6hog_expert/hog26-specialist2.pt" ] && exit 0
  rm -rf "$FL"
done
exit 1
