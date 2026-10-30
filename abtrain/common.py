"""Paths, settings, logging and small helpers shared by every stage.

Nothing here writes outside ``WORK`` (the network home) or ``SHM`` (/dev/shm): the server's
root disk is nearly full, so caches and temp files are pointed into ``WORK`` too.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("AB_WORK", str(Path.home() / "clashrl_ab"))).resolve()
_user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
_shm_default = Path("/dev/shm") / f"clashrl_ab_{_user}" if Path("/dev/shm").is_dir() else WORK / "shm"
SHM = Path(os.environ.get("AB_SHM", str(_shm_default)))

DATA = WORK / "data"
RAW = Path(os.environ.get("AB_RAW", str(DATA / "IL_Replay")))
SHARDS = DATA / "shards"
FL_SHARDS = DATA / "fl_shards"
RUNS = WORK / "runs"
LOGS = WORK / "logs"
LEAGUE = WORK / "league"
CKPT = WORK / "checkpoints"  # kept results: protected human-only stages, finals, best vs General
STATE = WORK / "state.json"
FL_REPO = Path(os.environ.get("AB_FL_REPO", str(WORK / "ext" / "FirstLight_CR")))
FL_PORT = REPO / "firstlight-port"

# IL_Replay on Hugging Face, pinned to the revision the earlier work used.
HF_DATASET = "VanguardX101/IL_Replay"
HF_REVISION = "059d43a02138a34b1b3009cc2acc7630fb99a638"

# Every tunable number in one place. ``WORK/config.json`` overrides any key (flat dict).
DEFAULTS: dict[str, Any] = {
    # wall clock: the whole pipeline stops cleanly this many hours after the first start
    "deadline_hours": 128.0,  # 66 h planned + 2 days added after the 24 Oct stop (60 GiB home quota)
    # A, phase 2 (Da 2026-10-26): PPO again from checkpoints/A/best_vs_general.pt (the 8M snapshot,
    # 0.73 vs FirstLight General) at delay 0, against the real FirstLight models, on all GPUs; B
    # paused. The first supervisor start with a_phase 2 archives phase 1's A run (WORK/archive) and
    # starts a new clock: the pipeline then stops a_phase_hours after that start.
    # A, phase 3 (Da 2026-10-30): PPO again from the best A, now with live-like handicaps on this
    # model's seats: it sees the state a3_obs_lag_ticks ago, its plays take a3_cmd_delay_ticks, and
    # each listed input is missing with probability a3_blind. Started and archived like phase 2.
    "a_phase": 3,
    "a_phase_hours": 72.0,
    "a3_obs_lag_ticks": 10,  # 500 ms, a multiple of the 5-tick decision window
    "a3_cmd_delay_ticks": 4,  # 200 ms
    "a3_blind": 0.5,
    "a3_eval_best_games": 60,  # vs the best A without any handicap
    "a3_eval_fl_games": 30,  # vs FirstLight General
    "run_B": False,  # B's PPO (paused in A's phase 2)
    "a_mix": {"fl_general": 0.2, "fl_hog2": 0.1, "self": 0.25, "past": 0.25, "bc": 0.1, "hard": 0.1},
    # stage 0 extraction
    "extract_workers": 46,
    "extract_limit_tasks": 0,  # tests only: 0 = every task
    "extract_limit_matches": 0,  # tests only: matches per task
    "matches_per_shard": 60,
    "holdout_mod": 50,  # match_hash % 50 == 0 -> validation (2% of matches)
    # BC (stage 1) and AWBC (stage 2)
    "bc_batch_per_gpu": 64,  # sequences per GPU per step
    "bc_window": 80,
    "bc_burn_in": 16,
    "bc_workers_per_gpu": 5,
    "bc_token_budget": 200_000,  # encoder tokens per forward; bigger batches are split
    "bc_max_steps": 0,  # tests only: cap optimizer steps of every supervised stage
    "bc1_passes": 1.0,
    "bc1_lr": 2e-4,
    "bc1_lr_end": 2e-5,
    "bc_warmup_steps": 1000,
    "bc_weight_decay": 0.01,
    "bc_grad_clip": 1.0,
    "gate_pos_weight": 3.0,
    "value_coef": 0.25,
    "value_subsample": 8,  # 1 in 8 states per match carry value loss
    "awbc_passes_A": 1.5,
    "stage2_A_all_gpus": True,  # A's stage 2 (B's starting point) on all GPUs before the split
    "awbc_lr": 1e-4,
    "awbc_lr_end": 1e-5,
    "awbc_n": 24,  # n-step look-ahead (decisions) for the advantage
    "awbc_beta": 1.0,
    "awbc_wmax": 20.0,
    "awbc_plain_frac": 0.3,  # loss = 0.3 plain BC + 0.7 weighted BC
    "awbc_ess_min": 0.3,
    "awbc_B_max_passes": 5,
    "awbc_B_general_frac": 0.3,
    "calib_brier3_max": 0.47,  # 3-class Brier; a constant 50/50 guess scores 0.5
    "calib_gap_max": 0.08,  # mean |predicted - observed| win rate over 10 bins
    "calib_conf_gain": 0.1,  # |P(win) - P(loss)| must be this much higher in the last 30 s than the first 10 s
    "abtest_games": 200,
    "abtest_min_winrate": 0.53,
    # FirstLight stand-ins (recorded through flport, cloned into our network)
    "fl_record_workers": 16,
    "fl_record_max_games": 1200,
    "standin_steps": 4000,
    "require_standins": True,  # PPO starts only once both FirstLight stand-ins are in the league
    "standin_wait_hours": 3.0,  # with require_standins false: how long PPO waits for them first
    "standin_lr": 5e-5,
    # PPO (stage 3-4)
    "ppo_workers": 40,  # at most this many simulator env worker processes (one core each) per run
    "ppo_fl_workers": 22,  # A phase 2: env workers (one core each) playing the real FirstLight
    # FirstLight battles per such worker. Every battle moves one decision per actor round (~1 s),
    # so 1 per worker gave FirstLight only ~3% of training; 6 x 22 ~ 30%
    "ppo_fl_envs_per_worker": 6,
    "ppo_envs_per_worker": 12,
    "ppo_segment": 160,  # 40 s of 250 ms decisions, LSTM state carried across segments
    "ppo_chunk": 64,
    "ppo_burn_in": 16,
    "ppo_minibatch_steps": 2048,
    "ppo_rollout_steps": 65536,  # learner decisions per rollout file (one PPO iteration)
    "ppo_epochs": 2,
    "ppo_clip": 0.1,
    "ppo_target_kl": 0.01,
    "ppo_lr": 1e-5,
    "ppo_gamma": 0.9997,
    "ppo_lambda": 0.98,
    "ppo_vf_coef": 0.5,
    "ppo_grad_clip": 0.5,
    "ppo_ent_start": 0.003,
    "ppo_ent_end": 0.001,
    "kl_ref_budget_start": 0.03,  # KL to the stage-2 model: a budget, kept low on purpose
    "kl_ref_budget_end": 0.1,
    "kl_ref_coef_init": 0.01,
    "kl_ref_coef_max": 0.05,
    "critic_warmup_iters": 2,
    "nominal_decisions_A": 64_000_000,
    "nominal_decisions_B": 48_000_000,
    "max_decisions_A": 2_000_000_000,  # in practice the deadline ends PPO
    "max_decisions_B": 2_000_000_000,
    "snapshot_every": 2_000_000,
    "keep_latest_snapshots": 3,  # per run, on disk: the newest snapshots ...
    "keep_best_snapshots": 2,  # ... plus the best by FirstLight screen (home quota is 60 GiB)
    "delay_ramp_frac": 0.2,
    "live_delay_ticks": 21,
    "ppo_fixed_delay_A": 0,  # ticks; None = phase 1's ramp to the live delay
    "ppo_fixed_delay_B": None,
    "b_bc_aux_coef": 0.2,
    "b_bc_aux_frac": 0.25,
    "b_elixir_coef": 0.01,
    "b_control_run": False,  # optional second B without the elixir-balance term (needs spare GPUs)
    "shaping_coef": 0.05,
    "overflow_personal": 0.001,
    "overflow_unilateral": 0.0005,
    "overflow_grace": 1.0,
    "overflow_cap": 0.1,
    # evaluation against the real FirstLight General through flport
    "fl_eval_workers": 6,
    "fl_eval_games": 80,  # screen: primary suite; the other suites get half each
    "fl_eval_decode": "argmax",  # one decoding for screens, promotion games and the best's score
    "fl_promo_games": 160,  # extra primary-suite games before a promotion
    "promote_gate": 0.6,  # minimum screen win rate vs General to try a promotion
    "standin_check_games": 20,
    "fl_eval_device": "cpu",
    "fl_eval_delay_ticks": 0,  # FirstLight's own benchmark runs without command delay
    "fl_baseline_games": 100,  # FL Hog2 vs FL General under the same protocol
    "target_winrate": 0.85,
    # robustness
    "stage_retries": 6,
    "checkpoint_minutes": 12,
    "hang_minutes": 40,
}


def settings() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    path = WORK / "config.json"
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text()))
        except Exception as exc:  # a broken override must not stop the run
            log(f"config.json ignored: {exc!r}")
    return cfg


def ensure_dirs() -> None:
    for p in (WORK, DATA, SHARDS, FL_SHARDS, RUNS, LOGS, LEAGUE, SHM, CKPT):
        p.mkdir(parents=True, exist_ok=True)


_LOG_NAME = os.environ.get("AB_LOG_NAME", "")


def log(msg: str, name: str | None = None) -> None:
    """Timestamped line to stdout and, when a name is set, to ``LOGS/<name>.log``."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{socket.gethostname()}:{os.getpid()}] {msg}"
    print(line, flush=True)
    name = name or _LOG_NAME
    if name:
        try:
            LOGS.mkdir(parents=True, exist_ok=True)
            with open(LOGS / f"{name}.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=str).encode())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def append_jsonl(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def torch_save_atomic(obj: Any, path: Path) -> None:
    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def deadline_ts() -> float:
    """Unix time at which everything should wrap up: fixed at the first start, or at the start
    of A's phase 2 (``a_phase_hours`` after it)."""
    st = read_json(STATE, {}) or {}
    cfg = settings()
    if st.get("a_phase_started") is not None:
        return float(st["a_phase_started"]) + cfg["a_phase_hours"] * 3600
    started = st.get("started_at")
    if started is None:
        return time.time() + cfg["deadline_hours"] * 3600
    return float(started) + cfg["deadline_hours"] * 3600


def hours_left() -> float:
    return (deadline_ts() - time.time()) / 3600.0


def die(msg: str, code: int = 1) -> None:
    log("FATAL " + msg)
    sys.exit(code)
