"""PPO with a league for one run (A or B) on two GPUs: an actor and a learner process.

    python -m abtrain.ppo --run A --actor-gpu 0 --learner-gpu 1 --cores 0-21 --init RUNS/A2/final.pt

Actor (GPU ``actor_gpu``): CPU env workers play battles; seats are batched by model (current
policy, frozen opponents, stand-ins) for inference. Learner seats are cut into 40 s segments
(160 decisions) with the LSTM state carried across them; GAE is computed per segment. The
reference model (the stage-2 checkpoint) runs alongside so each step also stores the reference
log-probability of the action taken, for the KL-to-reference term. Every ~64k learner decisions
a rollout file goes to /dev/shm.

Learner (GPU ``learner_gpu``): BPTT chunks of 64 decisions with 16 burn-in steps from stored
LSTM states, clip 0.1, 2 epochs, early stop at approx KL 1.5 x 0.01, AdamW lr 1e-5. First a
critic warm-up with the policy frozen. KL to the reference is a *budget*, not a big penalty:
the coefficient starts at 0.01, is capped at 0.05, adapts to keep the measured KL near a budget
that rises 0.03 -> 0.1 over the nominal run, and decays toward 0 after it (Da: no high KL
penalty). B adds a human Hog BC auxiliary loss (coef 0.2 -> 0 over the first 25%).

Checkpoints every ~12 min; restarting the same command resumes. League snapshots every 2M.
"""

from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from .common import (LEAGUE, RUNS, SHM, append_jsonl, atomic_write_json, deadline_ts, log, read_json,
                     settings, torch_load, torch_save_atomic)

_CTX = mp.get_context("spawn")
VALUE_MODULES = ("value", "v_res", "priv_mlp", "priv_lin", "priv_slot")


def run_dir(run: str) -> Path:
    return RUNS / f"ppo_{run}"


def shm_dir(run: str) -> Path:
    return SHM / f"ppo_{run}"


def nominal(run: str, cfg: dict) -> int:
    return cfg["nominal_decisions_A"] if run == "A" else cfg["nominal_decisions_B"]


def delay_for(run: str, decisions: int, cfg: dict, rng: np.random.Generator) -> int:
    """Command delay (ticks): ramps 0 -> live over the first 20% of the nominal run, then is
    randomised near the live 21 ticks. B's second half runs at live latency only."""
    nom = nominal(run, cfg)
    live = cfg["live_delay_ticks"]
    frac = decisions / max(1, nom)
    if frac < cfg["delay_ramp_frac"]:
        top = live * frac / cfg["delay_ramp_frac"]
        return int(round(rng.uniform(0, top)))
    if run == "B" and frac >= 0.5:
        return int(np.clip(round(rng.normal(live, 1.0)), live - 2, live + 2))
    if rng.random() < 0.8:
        return int(np.clip(round(rng.normal(live, 2.0)), 1, 25))
    return int(rng.integers(1, 26))


# =============================================================================== actor


class ModelCache:
    """Frozen opponents on the actor GPU (~30 MB each). Only models no running battle uses are
    evicted, so a large league never reloads the same checkpoints decision after decision."""

    def __init__(self, dev, cap: int = 64):
        self.dev, self.cap = dev, cap
        self.models: dict[str, object] = {}
        self.used: dict[str, float] = {}

    def get(self, key: str, path: str, active: set | None = None):
        from .model import build_model

        if key not in self.models:
            idle = [k for k in self.models if not active or k not in active]
            while len(self.models) >= self.cap and idle:
                old = min(idle, key=self.used.get)
                idle.remove(old)
                del self.models[old], self.used[old]
            m = build_model().to(self.dev)
            ck = torch_load(path)
            sd = ck["model"] if "model" in ck else ck
            m.load_state_dict({k: v.float() for k, v in sd.items()})
            m.eval()
            self.models[key] = m
        self.used[key] = time.time()
        return self.models[key]


class _Seg:
    __slots__ = ("rows", "labs", "logp", "logp_ref", "value", "reward", "states", "meta", "boot")

    def __init__(self, meta):
        self.rows, self.labs, self.logp, self.logp_ref, self.value, self.reward = [], [], [], [], [], []
        self.states = []  # (position, h, c, h_ref, c_ref) at 0 and 64k - 16
        self.meta = meta
        self.boot = False


def _gae(r, v, boot, done_last, gamma, lam):
    n = len(r)
    adv = np.zeros(n, np.float32)
    last = 0.0
    nxt = boot
    for t in range(n - 1, -1, -1):
        nonterm = 0.0 if (t == n - 1 and done_last) else 1.0
        delta = r[t] + gamma * nxt * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        nxt = v[t]
    return adv, adv + np.asarray(v, np.float32)


def actor_main(run: str, gpu: int, cores: list[int], init: str, stop):
    import torch

    from .env import BattleSpec, SeatSpec
    from .game import Catalog, make_engine
    from .league import League
    from .model import build_model
    from .shards import concat_sequences, pack_rows
    from .vec import VecBattles, batch_obs

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    cfg = settings()
    name = f"ppo_{run}_actor"
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    rd, sd = run_dir(run), shm_dir(run)
    rd.mkdir(parents=True, exist_ok=True)
    (sd / "rollouts").mkdir(parents=True, exist_ok=True)
    cat = Catalog.build(make_engine())
    seed = int(time.time()) % 100000
    league = League(run, cat, seed)
    rng = np.random.default_rng(seed + 17)
    rcfg = dict(gamma=cfg["ppo_gamma"], shaping=cfg["shaping_coef"], personal=cfg["overflow_personal"],
                unilateral=cfg["overflow_unilateral"], grace=cfg["overflow_grace"], cap=cfg["overflow_cap"],
                elixir_coef=cfg["b_elixir_coef"])
    n_workers = min(cfg["ppo_workers"], max(1, len(cores)))
    per = cfg["ppo_envs_per_worker"]
    vec = VecBattles(n_workers, per, rcfg, cores)
    nb = vec.n
    policy = build_model().to(dev).eval()
    ref = build_model().to(dev).eval()
    ck = torch_load(init)
    ref.load_state_dict(ck["model"] if "model" in ck else ck)
    cache = ModelCache(dev)
    version, decisions = -1, 0
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def load_weights():
        nonlocal version, decisions
        info = read_json(sd / "version.json", None)
        if not info or info["version"] == version:
            return False
        try:
            st = torch_load(sd / "weights.pt")
        except Exception:
            return False
        policy.load_state_dict(st)
        version, decisions = int(info["version"]), int(info["decisions"])
        return True

    while not load_weights():
        if stop.is_set():
            return
        time.sleep(2)
    log(f"actor up: {nb} battles on {n_workers} workers, weights v{version}", name)

    def make_spec():
        m = league.next_match()
        ls = m["learner_seat"]
        decks = [None, None]
        decks[ls] = league.decks.dealt(m["decks"][0])
        decks[1 - ls] = league.decks.dealt(m["decks"][1])
        learners = {0, 1} if m["opp"] == "self" else {ls}
        seats = tuple(SeatSpec(decks[s], elixir_shaping=(run == "B" and s in learners)) for s in (0, 1))
        delay = delay_for(run, decisions, cfg, rng)
        meta = {"opp": m["opp"], "learner_seat": ls, "learners": sorted(learners), "opp_deck": m["decks"][1],
                "own_deck": m["decks"][0], "delay": delay}
        return BattleSpec(seats, delay=delay, seed=int(rng.integers(2**62)), feature_drop=0.1, meta=meta)

    specs = [make_spec() for _ in range(nb)]
    obs = vec.reset(specs)
    metas = [s.meta for s in specs]
    nxt = [make_spec() for _ in range(nb)]
    core = policy.core_dim
    H = torch.zeros(nb, 2, core, device=dev)
    C = torch.zeros(nb, 2, core, device=dev)
    HR = torch.zeros(nb, 2, core, device=dev)
    CR = torch.zeros(nb, 2, core, device=dev)
    segs: dict[tuple[int, int], _Seg] = {}
    done_segs: list[_Seg] = []
    buf_steps = 0
    roll_n = int(time.time())
    seg_len, chunk, burn = cfg["ppo_segment"], cfg["ppo_chunk"], cfg["ppo_burn_in"]
    rollout_steps = cfg.get("ppo_rollout_steps", 65536)
    gamma, lam = cfg["ppo_gamma"], cfg["ppo_lambda"]
    ep_stats: list[dict] = []
    t_metric = t_league = t_hb = time.time()
    steps = 0

    def close_seg(key, seg: _Seg, boot: float, done: bool):
        nonlocal buf_steps
        if not seg.rows:
            return
        adv, ret = _gae(seg.reward, seg.value, boot, done, gamma, lam)
        seg.meta = dict(seg.meta, adv=adv, ret=ret)
        done_segs.append(seg)
        buf_steps += len(seg.rows)

    def flush_rollout():
        nonlocal done_segs, buf_steps, roll_n
        while len(glob.glob(str(sd / "rollouts" / "roll_*.npz"))) >= 2 and not stop.is_set():
            time.sleep(0.25)  # the learner is behind: keep policy lag small
        packed = [pack_rows(s.rows, s.labs) for s in done_segs]
        store = concat_sequences(packed)
        store["logp"] = np.concatenate([np.asarray(s.logp, np.float32) for s in done_segs])
        store["logp_ref"] = np.concatenate([np.asarray(s.logp_ref, np.float32) for s in done_segs])
        store["value"] = np.concatenate([np.asarray(s.value, np.float32) for s in done_segs])
        store["reward"] = np.concatenate([np.asarray(s.reward, np.float32) for s in done_segs])
        store["adv"] = np.concatenate([s.meta["adv"] for s in done_segs])
        store["ret"] = np.concatenate([s.meta["ret"] for s in done_segs])
        st_pos, st_h, st_c, st_hr, st_cr, st_off = [], [], [], [], [], [0]
        for s in done_segs:
            for pos, h, c, hr, cr in s.states:
                st_pos.append(pos)
                st_h.append(h), st_c.append(c), st_hr.append(hr), st_cr.append(cr)
            st_off.append(len(st_pos))
        store["st_pos"] = np.asarray(st_pos, np.int32)
        store["st_off"] = np.asarray(st_off, np.int64)
        for k, arr in (("st_h", st_h), ("st_c", st_c), ("st_hr", st_hr), ("st_cr", st_cr)):
            store[k] = np.stack(arr).astype(np.float16)
        store["version"] = np.int64(version)
        roll_n += 1
        path = sd / "rollouts" / f"roll_{roll_n:012d}.npz"
        tmp = sd / "rollouts" / f".tmp_{roll_n}.npz"
        with open(tmp, "wb") as fh:
            np.savez(fh, **store)
        os.replace(tmp, path)
        done_segs, buf_steps = [], 0

    while not stop.is_set():
        if steps % 20 == 0:
            load_weights()
        # ---- group seats by model and act
        groups: dict[str, list[tuple[int, int]]] = {}
        for k in range(nb):
            m = metas[k]
            for s in (0, 1):
                key = "cur" if (s in m["learners"]) else m["opp"]
                groups.setdefault(key, []).append((k, s))
        labs = [[None, None] for _ in range(nb)]
        learner_out = {}
        with torch.no_grad(), torch.autocast(**amp):
            active = set(groups)
            for key, seats in groups.items():
                if key == "cur":
                    net = policy
                else:
                    o = league.opps.get(key)
                    net = cache.get(key, o["path"], active) if o else policy
                ks = torch.tensor([k for k, _ in seats], device=dev)
                ss = torch.tensor([s for _, s in seats], device=dev)
                bt = batch_obs([obs[k][s] for k, s in seats], dev)
                st = (H[ks, ss][None].contiguous(), C[ks, ss][None].contiguous())
                lab, lp, v, _vl, (h2, c2) = net.act(bt, st)
                H[ks, ss], C[ks, ss] = h2[0].float(), c2[0].float()
                lab_np = lab.cpu().numpy()
                for i, (k, s) in enumerate(seats):
                    labs[k][s] = lab_np[i].tolist()
                if key == "cur":
                    # reference log-prob of the taken action, with the reference's own LSTM state
                    stR = (HR[ks, ss][None].contiguous(), CR[ks, ss][None].contiguous())
                    b1 = {kk: vv[:, None] for kk, vv in bt.items() if kk != "tiles"}
                    b1["lab"] = lab[:, None]
                    ar = torch.arange(len(seats), device=dev)
                    c1, c2_ = lab[:, 0], lab[:, 3]
                    b1["tm1"] = (bt["tiles"][ar, c1.clamp(0, 3)] & (c1 >= 0)[:, None] & (c1 < 4)[:, None])[:, None]
                    b1["tm2"] = (bt["tiles"][ar, c2_.clamp(0, 3)] & (c2_ >= 0)[:, None] & (c2_ < 4)[:, None])[:, None]
                    ro = ref.evaluate(b1, stR, need_value=False)
                    hR, cR = ro["state"]
                    old = (st[0][0].half().cpu().numpy(), st[1][0].half().cpu().numpy(),
                           stR[0][0].half().cpu().numpy(), stR[1][0].half().cpu().numpy())
                    HR[ks, ss], CR[ks, ss] = hR[0].float(), cR[0].float()
                    learner_out = {"seats": seats, "lp": lp.float().cpu().numpy(), "v": v.float().cpu().numpy(),
                                   "lpr": ro["logp"].float().cpu().numpy(), "old": old}
        # ---- record learner steps (and close segments waiting for their bootstrap value)
        if learner_out:
            for i, (k, s) in enumerate(learner_out["seats"]):
                seg = segs.get((k, s))
                if seg is not None and seg.boot:
                    close_seg((k, s), seg, float(learner_out["v"][i]), False)
                    seg = None
                if seg is None:
                    seg = _Seg(metas[k])
                    segs[(k, s)] = seg
                pos = len(seg.rows)
                if pos == 0 or (pos % chunk) == chunk - burn:
                    o = learner_out["old"]
                    seg.states.append((pos, o[0][i], o[1][i], o[2][i], o[3][i]))
                seg.rows.append(obs[k][s])
                seg.labs.append(labs[k][s])
                seg.logp.append(float(learner_out["lp"][i]))
                seg.logp_ref.append(float(learner_out["lpr"][i]))
                seg.value.append(float(learner_out["v"][i]))
        # ---- step every battle
        res = vec.step([(labs[k][0], labs[k][1]) for k in range(nb)], nxt)
        steps += 1
        for k, (o2, rew, done, info) in enumerate(res):
            m = metas[k]
            for s in m["learners"]:
                seg = segs.get((k, s))
                if seg is not None and seg.rows and len(seg.reward) < len(seg.rows):
                    seg.reward.append(float(rew[s]))
            if done:
                for s in m["learners"]:
                    seg = segs.pop((k, s), None)
                    if seg is not None:
                        close_seg((k, s), seg, 0.0, True)
                ls = m["learner_seat"]
                score = (info["result"][ls] + 1) / 2.0
                if m["opp"] != "self":
                    league.report(m["opp"], m["opp_deck"], score)
                ep_stats.append({"opp": m["opp"], "score": score, "ticks": info["ticks"],
                                 "plays": info["plays"][ls], "opp_plays": info["plays"][1 - ls],
                                 "first_play": info["first_play"][ls], "overflow": info["overflow"][ls],
                                 "trade": info["elixir_trade"][ls], "delay": m.get("delay", 0)})
                H[k] = 0
                C[k] = 0
                HR[k] = 0
                CR[k] = 0
                metas[k] = nxt[k].meta
                nxt[k] = make_spec()
            else:
                for s in m["learners"]:
                    seg = segs.get((k, s))
                    if seg is not None and len(seg.rows) >= seg_len:
                        seg.boot = True
            obs[k] = o2
        if buf_steps >= rollout_steps:
            flush_rollout()
        now = time.time()
        if now - t_hb > 30:
            t_hb = now
            (rd / "actor_heartbeat").write_text(str(now))
        if now - t_metric > 300 and ep_stats:
            t_metric = now
            append_jsonl(rd / "actor_metrics.jsonl", _episode_summary(ep_stats, version, decisions))
            ep_stats = []
        if now - t_league > 120:
            t_league = now
            league.save()
            atomic_write_json(rd / "league_summary.json", {"time": now, "decisions": decisions,
                                                          "opponents": league.summary()})
    vec.close()


def _episode_summary(eps: list[dict], version: int, decisions: int) -> dict:
    by: dict[str, list[float]] = {}
    for e in eps:
        kind = e["opp"].split(":")[0] if e["opp"] != "self" else "self"
        if e["opp"].startswith("standin:"):
            kind = e["opp"]
        by.setdefault(kind, []).append(e["score"])
    wins = [e for e in eps if e["score"] == 1.0]
    passive = sum(1 for e in wins if e["opp_plays"] <= 3)
    first = [e["first_play"] * 0.05 for e in eps if e["first_play"] >= 0]
    return {
        "time": time.time(), "version": version, "decisions": decisions, "episodes": len(eps),
        "score_by_opponent": {k: (round(float(np.mean(v)), 3), len(v)) for k, v in by.items()},
        "plays_per_game": float(np.mean([e["plays"] for e in eps])),
        "first_play_s": float(np.mean(first)) if first else None,
        "wins_vs_passive_share": passive / max(1, len(wins)),
        "overflow_elixir": float(np.mean([e["overflow"] for e in eps])),
        "elixir_trade": float(np.mean([e["trade"] for e in eps])),
        "game_s": float(np.mean([e["ticks"] for e in eps])) * 0.05,
        "delay_ticks": float(np.mean([e["delay"] for e in eps])),
    }


# =============================================================================== learner


def _chunks(store: dict, chunk: int, burn: int):
    """(start, end, loss_from, loss_to, state_row) per BPTT chunk of every segment."""
    out = []
    off, st_off, st_pos = store["seq_off"], store["st_off"], store["st_pos"]
    for s in range(len(off) - 1):
        a, b = int(off[s]), int(off[s + 1])
        rows = {int(st_pos[r]): r for r in range(int(st_off[s]), int(st_off[s + 1]))}
        for j, ls in enumerate(range(a, b, chunk)):
            pos = 0 if j == 0 else (ls - a) - burn
            if pos not in rows:
                continue
            out.append((a + pos, b, ls, min(b, ls + chunk), rows[pos]))
    return out


def learner_main(run: str, gpu: int, init: str, stop):
    import torch

    from .bc import WindowStream, bc_losses, data_root, make_loader, micro_splits
    from .model import build_model, to_device_batch
    from .shards import collate

    cfg = settings()
    name = f"ppo_{run}_learner"
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    rd, sd = run_dir(run), shm_dir(run)
    rd.mkdir(parents=True, exist_ok=True)
    (sd / "rollouts").mkdir(parents=True, exist_ok=True)
    (LEAGUE / run).mkdir(parents=True, exist_ok=True)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ppo_lr"], eps=1e-5, weight_decay=0.0)
    latest = rd / "latest.pt"
    st = {"decisions": 0, "iteration": 0, "kl_coef": cfg["kl_ref_coef_init"], "version": 0,
          "warmup_done": 0, "ev_hist": [], "next_snapshot": cfg["snapshot_every"],
          "kl_budget": cfg["kl_ref_budget_start"]}
    ck = None
    for cand in (latest, rd / "latest.new.pt", rd / "prev.pt"):  # newest loadable checkpoint first
        if cand.exists():
            try:
                ck = torch_load(cand)
                break
            except Exception as exc:
                log(f"checkpoint {cand.name} unreadable ({exc!r}); trying an older one", name)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st.update(ck["state"])
        log(f"resumed at {st['decisions']} decisions, iteration {st['iteration']}", name)
    else:
        ck = torch_load(init)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        log(f"initialised from {init}", name)
    nom = nominal(run, cfg)
    max_dec = cfg["max_decisions_A"] if run == "A" else cfg["max_decisions_B"]
    token_budget = cfg["bc_token_budget"]
    deadline = deadline_ts()
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def publish():
        st["version"] += 1
        sdict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch_save_atomic(sdict, sd / "weights.pt")
        atomic_write_json(sd / "version.json", {"version": st["version"], "decisions": st["decisions"]})

    def checkpoint(final=False):
        # write the new one completely first, then rotate: some loadable file always exists
        new = rd / "latest.new.pt"
        torch_save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, new)
        if latest.exists():
            os.replace(latest, rd / "prev.pt")
        os.replace(new, latest)
        if final:
            torch_save_atomic({"model": model.state_dict(), "decisions": st["decisions"]}, rd / "final.pt")

    def snapshot():
        tag = f"{st['decisions']:012d}"
        sdict = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        torch_save_atomic({"model": sdict, "decisions": st["decisions"], "run": run}, LEAGUE / run / f"snap_{tag}.pt")
        log(f"league snapshot {tag}", name)

    for f in glob.glob(str(sd / "rollouts" / "*.npz")):  # stale rollouts from before a restart
        os.remove(f)
    publish()
    # B: human Hog windows for the BC auxiliary loss (first 25% of the nominal run)
    aux_iter = None
    if run == "B" and cfg["b_bc_aux_coef"] > 0 and st["decisions"] < cfg["b_bc_aux_frac"] * nom:
        try:
            from .common import SHARDS

            idx = read_json(SHARDS.parent / "index.json", {})
            hog = [s["path"] for s in idx.get("shards", []) if s["hog_steps"] > 0]
            if hog:
                ds = WindowStream(hog, data_root(), 0, 1, 16, 80, 16, "train", "hog", seed=st["iteration"] + 3)
                aux_iter = iter(make_loader(ds, 2))
        except Exception as exc:
            log(f"BC auxiliary data unavailable: {exc!r}", name)
    t_ck = time.time()
    value_names = [n for n, _ in model.named_parameters() if n.split(".")[0] in VALUE_MODULES]
    finished = False
    while not stop.is_set():
        if st["decisions"] >= max_dec or time.time() > deadline:
            log(f"stopping: decisions {st['decisions']}, deadline passed {time.time() > deadline}", name)
            finished = True
            break
        files = sorted(glob.glob(str(sd / "rollouts" / "roll_*.npz")))
        if not files:
            time.sleep(0.5)
            continue
        path = files[0]
        try:
            with np.load(path) as z:
                store = {k: z[k] for k in z.files}
        except Exception as exc:
            log(f"bad rollout {path}: {exc!r}", name)
            os.remove(path)
            continue
        os.remove(path)
        t_it = time.time()
        n_steps = len(store["lab"])
        chunks = _chunks(store, cfg["ppo_chunk"], cfg["ppo_burn_in"])
        if not chunks:
            continue
        warm = not st["warmup_done"]
        for n, p in model.named_parameters():
            p.requires_grad_(not warm or n in value_names)
        frac = st["decisions"] / max(1, nom)
        ent_coef = cfg["ppo_ent_start"] + (cfg["ppo_ent_end"] - cfg["ppo_ent_start"]) * min(1.0, frac)
        budget = st.setdefault("kl_budget", cfg["kl_ref_budget_start"])
        aux_coef = cfg["b_bc_aux_coef"] * max(0.0, 1.0 - frac / cfg["b_bc_aux_frac"]) if run == "B" else 0.0
        per_mb = max(1, cfg["ppo_minibatch_steps"] // cfg["ppo_chunk"])
        ag: dict[str, list[float]] = {}
        rng = np.random.default_rng(st["iteration"])
        adv_all = store["adv"]
        stop_early = interrupted = False
        for epoch in range(cfg["ppo_epochs"]):
            order = rng.permutation(len(chunks))
            for i in range(0, len(order), per_mb):
                if stop.is_set():  # stop requested: checkpoint now rather than finish the iteration
                    interrupted = True
                    break
                sel = [chunks[j] for j in order[i : i + per_mb]]
                starts = np.array([c[0] for c in sel], np.int64)
                ends = np.array([c[1] for c in sel], np.int64)
                lfrom = np.array([c[2] for c in sel], np.int64)
                lto = np.array([c[3] for c in sel], np.int64)
                rows = np.array([c[4] for c in sel], np.int64)
                L = int((np.minimum(ends, lto) - starts).max())
                b = collate(store, starts, L, ends, lfrom, extra={k: store[k] for k in ("logp", "logp_ref", "adv", "ret", "value")},
                            loss_to=lto)
                b = to_device_batch(b, dev)
                b["h0"] = torch.from_numpy(store["st_h"][rows]).float().to(dev)
                b["c0"] = torch.from_numpy(store["st_c"][rows]).float().to(dev)
                lm_all = b["loss"]
                nl = lm_all.sum().clamp_min(1).float()
                adv_all = b["adv"].float()
                mu = (adv_all * lm_all).sum() / nl
                sdv = (((adv_all - mu) ** 2) * lm_all).sum().div(nl).sqrt()
                b["adv_n"] = (adv_all - mu) / (sdv + 1e-8)
                opt.zero_grad(set_to_none=True)
                sums = dict.fromkeys(("pg", "vf", "ent", "kl_ref", "approx_kl", "clipfrac", "r1", "r2", "res"), 0.0)
                finite = True
                # gradient accumulation over parts under the token budget: one late-game state
                # with many units pads the whole minibatch, so a single forward could OOM
                for sub in micro_splits(b, token_budget):
                    with torch.autocast(**amp):
                        o = model.evaluate(sub, (sub["h0"][None].contiguous(), sub["c0"][None].contiguous()))
                    lm = sub["loss"].reshape(-1)
                    adv = sub["adv_n"].reshape(-1)
                    logp = o["logp"].float()
                    lr_ = (logp - sub["logp"].reshape(-1)).clamp(-20, 20)
                    ratio = torch.exp(lr_)
                    pg = (-torch.minimum(ratio * adv, ratio.clamp(1 - cfg["ppo_clip"], 1 + cfg["ppo_clip"]) * adv) * lm).sum()
                    ret = sub["ret"].reshape(-1).float()
                    v = o["v"].float()
                    vf = (0.5 * (v - ret) ** 2 * lm).sum()
                    ent = (o["entropy"].float() * lm).sum()
                    d_ref = (sub["logp_ref"].reshape(-1) - logp).clamp(-20, 20)
                    kl_ref = ((torch.exp(d_ref) - 1 - d_ref) * lm).sum()
                    if warm:
                        loss = cfg["ppo_vf_coef"] * vf / nl
                    else:
                        loss = (pg + cfg["ppo_vf_coef"] * vf - ent_coef * ent + st["kl_coef"] * kl_ref) / nl
                    if not torch.isfinite(loss):
                        finite = False
                        break
                    loss.backward()
                    with torch.no_grad():
                        for k_, v_ in (("pg", pg), ("vf", vf), ("ent", ent), ("kl_ref", kl_ref),
                                       ("approx_kl", ((torch.exp(lr_) - 1 - lr_) * lm).sum()),
                                       ("clipfrac", (((ratio - 1).abs() > cfg["ppo_clip"]).float() * lm).sum()),
                                       ("r1", (ret * lm).sum()), ("r2", (ret * ret * lm).sum()),
                                       ("res", ((ret - v) ** 2 * lm).sum())):
                            sums[k_] += float(v_)
                    del o, loss
                if finite and not warm and aux_iter is not None and aux_coef > 0:
                    # B: human Hog BC auxiliary loss, its own backward so both graphs are never alive at once
                    hb = to_device_batch(next(aux_iter), dev)
                    n_h = hb["glob"].shape[0]
                    for sub in micro_splits(hb, token_budget):
                        with torch.autocast(**amp):
                            ho = model.evaluate(sub, need_value=False)
                        aux = aux_coef * bc_losses(ho, sub, cfg["gate_pos_weight"])["total"] * (sub["glob"].shape[0] / n_h)
                        if not torch.isfinite(aux):
                            finite = False
                            break
                        aux.backward()
                        sums["aux"] = sums.get("aux", 0.0) + float(aux)
                        del ho, aux
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["ppo_grad_clip"])
                if not finite or not torch.isfinite(gn):
                    _a(ag, "skipped", 1.0)
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                n = float(nl)
                mean_r = sums["r1"] / n
                var_r = max(0.0, sums["r2"] / n - mean_r * mean_r)
                approx_kl = sums["approx_kl"] / n
                for k_ in ("pg", "vf", "ent", "kl_ref", "clipfrac"):
                    _a(ag, k_, sums[k_] / n)
                _a(ag, "approx_kl", approx_kl)
                _a(ag, "ev", 1.0 - (sums["res"] / n) / (var_r + 1e-8))
                _a(ag, "grad_norm", float(gn))
                if "aux" in sums:
                    _a(ag, "aux_bc", sums["aux"] / max(aux_coef, 1e-9))
                if not warm and approx_kl > 1.5 * cfg["ppo_target_kl"]:
                    stop_early = True
                    break
            if stop_early or interrupted:
                break
        for p in model.parameters():
            p.requires_grad_(True)
        if interrupted:
            break
        means = {k: float(np.mean(v)) for k, v in ag.items()}
        st["iteration"] += 1
        if warm:
            st["ev_hist"].append(means.get("ev", 0.0))
            h = st["ev_hist"]
            need = max(1, cfg["critic_warmup_iters"])
            stable = len(h) >= 2 and abs(h[-1] - h[-2]) < 0.02
            if len(h) >= need and (stable or len(h) >= 3 * need):
                st["warmup_done"] = 1
                log(f"critic warm-up done after {len(h)} iterations (explained variance {h[-1]:.3f})", name)
        else:
            st["decisions"] += n_steps
            # the KL budget is relaxed toward its end value only while external evals improve
            if eval_improving(run):
                step_b = (cfg["kl_ref_budget_end"] - cfg["kl_ref_budget_start"]) * n_steps / max(1, nom)
                st["kl_budget"] = min(cfg["kl_ref_budget_end"], st["kl_budget"] + step_b)
            kl = means.get("kl_ref", 0.0)
            if st["decisions"] > nom:  # past the nominal run: let the anchor fade out
                st["kl_coef"] *= 0.9
            elif kl > 1.5 * budget:
                st["kl_coef"] = min(cfg["kl_ref_coef_max"], st["kl_coef"] * 1.5)
            elif kl < budget / 1.5:
                st["kl_coef"] = max(1e-4, st["kl_coef"] / 1.5)
        publish()
        rec = {"time": time.time(), "iteration": st["iteration"], "decisions": st["decisions"], "warmup": warm,
               "steps": n_steps, "seconds": time.time() - t_it, "kl_coef": st["kl_coef"], "kl_budget": budget,
               "ent_coef": ent_coef, "aux_coef": aux_coef, "early_stop": stop_early,
               "rollout_version": int(store["version"]), "version": st["version"],
               "reward_mean": float(store["reward"].mean()), **means}
        append_jsonl(rd / "learner_metrics.jsonl", rec)
        if st["iteration"] % 10 == 0:
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if k != "time"), name)
        (rd / "heartbeat").write_text(str(time.time()))
        if st["decisions"] >= st["next_snapshot"]:
            snapshot()
            st["next_snapshot"] += cfg["snapshot_every"]
        if time.time() - t_ck > cfg["checkpoint_minutes"] * 60:
            t_ck = time.time()
            checkpoint()
    if not finished:  # stopped from outside (SIGTERM, actor crash): resume later
        checkpoint()
        log(f"stopped at {st['decisions']} decisions; checkpoint written, will resume", name)
        return
    checkpoint(final=True)
    snapshot()
    if st["decisions"] >= max_dec:
        (rd / "DONE").write_text(str(st["decisions"]))
        log(f"finished at {st['decisions']} decisions", name)
    else:  # a rerun with a later deadline (config deadline_hours) continues from here
        atomic_write_json(rd / "DEADLINE_STOP", {"decisions": st["decisions"], "deadline": deadline})
        log(f"deadline reached at {st['decisions']} decisions; final weights written", name)


def _a(ag, k, v):
    ag.setdefault(k, []).append(v)


def run_finished(run: str) -> bool:
    """DONE (decision budget used up), or stopped at the deadline that still applies."""
    rd = run_dir(run)
    if (rd / "DONE").exists():
        return True
    stop = read_json(rd / "DEADLINE_STOP", None)
    return bool(stop) and time.time() >= deadline_ts() - 60


def eval_improving(run: str) -> bool:
    """True while the FirstLight evaluator reports improvement (or has not reported yet)."""
    st = read_json(RUNS / "eval" / f"{run}_state.json", None)
    return True if not st else bool(st.get("improving", True))


# =============================================================================== main


def _parse_cores(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _guard(stop, parent: int) -> None:
    """In the actor and learner: SIGTERM/SIGINT to the whole process group must not kill them
    before the learner checkpoints (the coordinator sets ``stop``); if the coordinator dies,
    stop on our own."""
    import threading

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

    def watch():
        while not stop.is_set():
            if os.getppid() != parent:
                stop.set()
            time.sleep(5)

    threading.Thread(target=watch, daemon=True).start()


def _actor_entry(run, gpu, cores, init, stop, parent):
    _guard(stop, parent)
    try:
        actor_main(run, gpu, cores, init, stop)
    except Exception:
        log("actor crashed:\n" + traceback.format_exc(), f"ppo_{run}_actor")
        raise


def _learner_entry(run, gpu, init, stop, parent):
    _guard(stop, parent)
    try:
        learner_main(run, gpu, init, stop)
    except Exception:
        log("learner crashed:\n" + traceback.format_exc(), f"ppo_{run}_learner")
        raise


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["A", "B"])
    ap.add_argument("--actor-gpu", type=int, default=0)
    ap.add_argument("--learner-gpu", type=int, default=1)
    ap.add_argument("--cores", default="0-21")
    ap.add_argument("--init", required=True, help="stage-2 checkpoint: PPO start and KL reference")
    a = ap.parse_args(argv)
    if run_finished(a.run):
        log(f"run {a.run} already finished")
        return
    (run_dir(a.run) / "DEADLINE_STOP").unlink(missing_ok=True)
    stop = _CTX.Event()
    cores = _parse_cores(a.cores)
    learner = _CTX.Process(target=_learner_entry, args=(a.run, a.learner_gpu, a.init, stop, os.getpid()))
    actor = _CTX.Process(target=_actor_entry, args=(a.run, a.actor_gpu, cores, a.init, stop, os.getpid()))
    learner.start()
    actor.start()

    def _term(*_):
        stop.set()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    code = 0
    while True:
        time.sleep(5)
        if stop.is_set():  # asked to stop: not a failure, whatever exits first
            break
        if not learner.is_alive():
            code = learner.exitcode or 0
            break
        if not actor.is_alive():
            code = actor.exitcode or 1  # the actor never ends on its own
            break
    stop.set()
    for p, wait in ((actor, 120), (learner, 600)):  # the learner writes its checkpoint first
        p.join(timeout=wait)
        if p.is_alive():
            p.kill()
    if code:
        raise SystemExit(f"run {a.run} stopped with code {code}")


if __name__ == "__main__":
    main()
