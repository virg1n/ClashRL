#!/usr/bin/env bash
# The one command for the A/B server:
#
#   bash run_ab.sh            # set up (first time), then train A and B unattended in the background
#   bash run_ab.sh status     # progress report (WORK/STATUS.md)
#   bash run_ab.sh stop       # stop everything cleanly; rerun `bash run_ab.sh` to resume
#   bash run_ab.sh fg         # same as the default, in the foreground
#
# Rerunning after a crash, a stopped Jupyter session or a reboot resumes where it left off.
# Everything lives in $AB_WORK (default ~/clashrl_ab); logs in $AB_WORK/logs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AB_WORK="${AB_WORK:-$HOME/clashrl_ab}"
source "$REPO/abtrain/env.sh"
PY="$AB_WORK/venv/bin/python"
cd "$REPO"

case "${1:-start}" in
  status)
    exec "$PY" -m abtrain.pipeline status ;;
  stop)
    exec "$PY" -m abtrain.pipeline stop ;;
  start|fg) ;;
  *) echo "usage: bash run_ab.sh [start|status|stop|fg]"; exit 2 ;;
esac

bash "$REPO/setup_ab.sh"
rm -f "$AB_WORK/STOP"

# Keeps the supervisor alive: it exits 0 when every lane has finished, 3 when stopped, 5 when
# it finished but a requirement was never met (see STATUS.md); 4 (a required stage gave up) and
# any other exit (a crash) restart it, and it resumes from its state file.
loop() {
  local restarts=0
  while true; do
    set +e
    "$PY" -m abtrain.pipeline run
    code=$?
    set -e
    [ "$code" = 0 ] && { echo "$(date '+%F %T') pipeline finished"; return 0; }
    [ -f "$AB_WORK/STOP" ] || [ "$code" = 3 ] && { echo "$(date '+%F %T') pipeline stopped"; return 3; }
    [ "$code" = 5 ] && { echo "$(date '+%F %T') pipeline FINISHED WITH UNMET REQUIREMENTS: see $AB_WORK/STATUS.md"; return 5; }
    restarts=$((restarts + 1))
    if [ "$restarts" -gt 40 ]; then echo "$(date '+%F %T') too many restarts, giving up"; return "$code"; fi
    wait_s=60; [ "$code" = 4 ] && wait_s=1200   # main lane gave up: wait before trying again
    echo "$(date '+%F %T') supervisor exited with $code; restart $restarts in ${wait_s}s"
    sleep "$wait_s"
  done
}

if [ "${1:-start}" = fg ]; then
  set +e
  loop 2>&1 | tee -a "$AB_WORK/logs/supervisor.out"
  exit "${PIPESTATUS[0]}"
fi

if [ -f "$AB_WORK/supervisor.json" ] && "$PY" - <<'EOF'
import json, os, socket, sys, time
p = os.path.join(os.environ["AB_WORK"], "supervisor.json")
try:
    d = json.load(open(p))
    os.kill(int(d["pid"]), 0)
    alive = d.get("host") == socket.gethostname() and time.time() - d.get("time", 0) < 600
except Exception:
    alive = False
sys.exit(0 if alive else 1)
EOF
then
  echo "already running; progress: bash run_ab.sh status"
  exit 0
fi

export -f loop
export PY
SETSID=""; command -v setsid >/dev/null 2>&1 && SETSID=setsid
nohup $SETSID bash -c loop >> "$AB_WORK/logs/supervisor.out" 2>&1 < /dev/null &
echo "started in the background (pid $!)."
echo "  progress:  bash run_ab.sh status     (or: cat $AB_WORK/STATUS.md)"
echo "  logs:      $AB_WORK/logs/"
echo "  stop:      bash run_ab.sh stop       (rerun bash run_ab.sh to resume)"
