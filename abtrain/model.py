"""AtlasNet (~7M): entity transformer + spatial block + LSTM, FirstLight-V4-style action head,
and a win/draw/loss value head with a privileged branch.

Batch layout (numpy or torch, leading dims [B, T]):
  glob [G] f, cards [16] long (card id + 1), cardf [16, CF] f, ent_ids [E, 3] long,
  ent_f [E, F] f, ent_mask [E] bool (True = real), priv_ids [13] long, priv_f [PF] f,
  cand [7] bool (legal candidates), ccost [7] f (elixir cost), tiles [4, 576] bool (act) or
  tm1 / tm2 [576] bool (tile masks of the labelled / taken first and second candidates).
Actions ``lab`` [6] long: c1, t1, d1, c2, t2, d2, -1 = none.

Privileged inputs (opponent hand, next card, deck, exact elixir) only reach the value head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .game import (CARD_ROWS, CARD_VOCAB, CF, F, G, HAND, N_KINDS, NBTN, NCAND, NDELAY, NT, NX, NY, PF,
                   PRIV_IDS, SECOND_PLAY_SLACK, UNIT_VOCAB)

NEG = -1e9
EXPLICIT_CH = 12
_TGT_COLS = [16, 17, 18, 19, 20, 39, 40, 41, 45]


def _mlp(i: int, h: int, o: int, drop: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(i, h), nn.GELU()]
    if drop:
        layers.append(nn.Dropout(drop))
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        y = self.c1(Fn.gelu(self.n1(x)))
        y = self.c2(Fn.gelu(self.n2(y)))
        return x + y


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax over legal entries; rows with no legal entry give 0 everywhere (unused)."""
    any_ = mask.any(-1, keepdim=True)
    m = mask | ~any_
    return torch.log_softmax(logits.float().masked_fill(~m, NEG), -1)


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return -(p * logp).masked_fill(~mask, 0.0).sum(-1)


class AtlasNet(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, ffn: int = 1024, core: int = 512,
                 sp_ch: int = 64, sp_proj: int = 32, tile_k: int = 128):
        super().__init__()
        self.d, self.core_dim, self.tile_k = d, core, tile_k
        # tokens
        self.ent_lin = nn.Linear(F, d)
        self.emb_unit = nn.Embedding(UNIT_VOCAB, d)
        self.emb_card = nn.Embedding(CARD_VOCAB, d)
        self.emb_kind = nn.Embedding(N_KINDS, d)
        self.card_lin = nn.Linear(CF, d)
        self.card_slot = nn.Embedding(CARD_ROWS, d)
        self.btn_emb = nn.Embedding(NBTN, d)
        self.btn_lin = nn.Linear(3, d)
        self.glob_mlp = _mlp(G, d, d)
        self.tok_norm = nn.LayerNorm(d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        # spatial
        self.sp_proj = nn.Linear(d, sp_proj)
        self.sp_in = nn.Conv2d(sp_proj + EXPLICIT_CH + 2, sp_ch, 3, padding=1)
        self.sp_blocks = nn.Sequential(*[ResBlock(sp_ch) for _ in range(3)])
        self.sp_pool = nn.Linear(2 * sp_ch, 128)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, NY), torch.linspace(0, 1, NX), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None], persistent=False)
        # core
        self.fuse = nn.Sequential(nn.Linear(d + d + 128, core), nn.GELU())
        self.lstm = nn.LSTM(core, core, batch_first=True)
        # policy head
        self.ctx = _mlp(core + d, d, d)
        self.gate = nn.Linear(d, 1)
        self.q1 = nn.Linear(d, d)
        self.kc = nn.Linear(d, d)
        self.tile_pos = nn.Parameter(torch.zeros(NT, sp_ch))
        self.tile_key = nn.Linear(sp_ch, tile_k)
        self.tq = _mlp(2 * d, d, tile_k)
        self.delay = _mlp(2 * d + tile_k, d // 2, NDELAY)
        self.delay_emb = nn.Embedding(NDELAY, d)
        self.tile_to_d = nn.Linear(tile_k, d)
        self.cont = _mlp(2 * d, d // 2, 1)
        self.shadow = _mlp(d + 2, d, d)
        self.q2 = nn.Linear(d, d)
        # value head (privileged branch: value only)
        self.priv_slot = nn.Embedding(PRIV_IDS, d)
        self.priv_lin = nn.Linear(PF, d)
        self.priv_mlp = _mlp(d, d, 128)
        self.value = nn.Sequential(nn.Linear(core + 128, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3))
        self.v_res = nn.Sequential(nn.Linear(core + 128, 128), nn.GELU(), nn.Linear(128, 1))
        self._init()

    def _init(self):
        nn.init.constant_(self.gate.bias, _logit(0.15))
        nn.init.normal_(self.gate.weight, std=0.01)
        last = self.cont[-1]
        nn.init.constant_(last.bias, _logit(0.25))
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(self.v_res[-1].weight)
        nn.init.zeros_(self.v_res[-1].bias)
        nn.init.normal_(self.tile_pos, std=0.02)

    # ------------------------------------------------------------------ trunk
    def initial_state(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, n, self.core_dim, device=device)
        return (z, z.clone())

    def _encode(self, b: dict) -> dict:
        """Per-step encoding of N flattened steps."""
        glob, cards, cardf = b["glob"], b["cards"], b["cardf"]
        ent_ids, ent_f, ent_mask = b["ent_ids"], b["ent_f"], b["ent_mask"]
        n = glob.shape[0]
        e = ent_f.shape[1]
        ent_tok = (self.ent_lin(ent_f) + self.emb_unit(ent_ids[..., 0]) + self.emb_card(ent_ids[..., 1])
                   + self.emb_kind(ent_ids[..., 2]))
        slots = torch.arange(CARD_ROWS, device=glob.device)
        card_tok = self.emb_card(cards) + self.card_lin(cardf) + self.card_slot(slots)[None]
        bfeat = torch.stack([glob[:, [22, 26, 29]], glob[:, [23, 27, 30]], glob[:, [25, 28, 31]]], -1)
        btn_tok = self.btn_emb(torch.arange(NBTN, device=glob.device))[None] + self.btn_lin(bfeat)
        glob_tok = self.glob_mlp(glob)[:, None]
        tok = self.tok_norm(torch.cat([glob_tok, card_tok, btn_tok, ent_tok], 1))
        pad = torch.cat([torch.zeros(n, 1 + CARD_ROWS + NBTN, dtype=torch.bool, device=glob.device), ~ent_mask], 1)
        out = self.enc_norm(self.encoder(tok, src_key_padding_mask=pad))
        g_out = out[:, 0]
        mem = torch.cat([out[:, 1 : 1 + HAND], out[:, 1 + CARD_ROWS : 1 + CARD_ROWS + NBTN]], 1)  # [N,7,d]
        ent_out = out[:, 1 + CARD_ROWS + NBTN :]
        em = ent_mask.unsqueeze(-1).to(ent_out.dtype)
        ent_mean = (ent_out * em).sum(1) / em.sum(1).clamp_min(1.0)
        # spatial scatter of entity tokens onto the 18 x 32 grid
        tx = (ent_f[..., 0] * NX).long().clamp(0, NX - 1)
        ty = (ent_f[..., 1] * NY).long().clamp(0, NY - 1)
        idx = (ty * NX + tx).masked_fill(~ent_mask, 0)
        proj = self.sp_proj(ent_out) * em
        grid = torch.zeros(n, NT, proj.shape[-1], device=glob.device, dtype=proj.dtype)
        grid.scatter_add_(1, idx[..., None].expand_as(proj), proj)
        ex = self._explicit(ent_f, ent_ids, ent_mask, idx).to(grid.dtype)
        x = torch.cat([grid, ex], -1).transpose(1, 2).reshape(n, -1, NY, NX)
        x = torch.cat([x, self.coords.expand(n, -1, -1, -1).to(x.dtype)], 1)
        x = self.sp_blocks(self.sp_in(x))
        sp_pool = self.sp_pool(torch.cat([x.mean((2, 3)), x.amax((2, 3))], -1))
        tiles = x.flatten(2).transpose(1, 2) + self.tile_pos[None].to(x.dtype)  # [N,576,ch]
        fused = self.fuse(torch.cat([g_out, ent_mean, sp_pool], -1))
        return {"fused": fused, "g": g_out, "mem": mem, "tiles": tiles}

    def _explicit(self, ent_f, ent_ids, ent_mask, idx):
        n, e, _ = ent_f.shape
        own = ent_f[..., 6] > 0.5
        air = ent_f[..., 7] > 0.5
        kind = ent_ids[..., 2]
        troop = (kind == 0) & ent_mask
        bld = (kind == 1) & ent_mask
        spell = (kind == 4) & ent_mask
        proj = (kind == 5) & ent_mask
        hp = ent_f[..., 2] * ent_f[..., 4]
        chans = [
            (troop & own & ~air).float(), (troop & ~own & ~air).float(), (troop & own & air).float(),
            (troop & ~own & air).float(), (bld & own).float(), (bld & ~own).float(),
            hp * (troop & own).float(), hp * (troop & ~own).float(),
        ]
        out = torch.zeros(n, NT, EXPLICIT_CH, device=ent_f.device)
        for c, v in enumerate(chans):
            out[..., c].scatter_add_(1, idx, v.float())
        # where spells will land and where projectiles are heading
        for c, (sel, dx, dy) in enumerate(((spell & own, 35, 36), (spell & ~own, 35, 36),
                                           (proj & own, 35, 36), (proj & ~own, 35, 36))):
            ax = ((ent_f[..., 0] + ent_f[..., dx]) * NX).long().clamp(0, NX - 1)
            ay = ((ent_f[..., 1] + ent_f[..., dy]) * NY).long().clamp(0, NY - 1)
            out[..., 8 + c].scatter_add_(1, (ay * NX + ax).masked_fill(~sel, 0), sel.float())
        return out

    def trunk(self, b: dict, state) -> tuple[dict, tuple]:
        """b has [B, T, ...] tensors. Returns per-step features (flattened N = B*T) and state."""
        bsz, t = b["glob"].shape[:2]
        flat = {k: b[k].reshape(bsz * t, *b[k].shape[2:]) for k in
                ("glob", "cards", "cardf", "ent_ids", "ent_f", "ent_mask")}
        enc = self._encode(flat)
        x = enc["fused"].reshape(bsz, t, -1).float()
        if state is None:
            state = self.initial_state(bsz, x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            h, new_state = self.lstm(x, (state[0].float(), state[1].float()))
        enc["h"] = h.reshape(bsz * t, -1)
        return enc, new_state

    # ------------------------------------------------------------------ heads
    def _ctx(self, enc):
        return self.ctx(torch.cat([enc["h"].to(enc["g"].dtype), enc["g"]], -1))

    def _cand_logits(self, q, mem):
        return torch.einsum("nd,nkd->nk", q, self.kc(mem)) / math.sqrt(self.d)

    def _tile_logits(self, ctx, mem_c, tiles):
        q = self.tq(torch.cat([ctx, mem_c], -1))
        keys = self.tile_key(tiles)
        return torch.einsum("nk,ntk->nt", q, keys) / math.sqrt(self.tile_k), keys

    def _gather(self, x, i):
        """x [N, K, ...], i [N] (>= 0) -> x[n, i[n]]."""
        return x[torch.arange(x.shape[0], device=x.device), i]

    def _act_emb(self, mem_c, tile_key_t, d):
        return mem_c + self.tile_to_d(tile_key_t) + self.delay_emb(d)

    def value_out(self, enc, b_flat) -> tuple[torch.Tensor, torch.Tensor]:
        pid = b_flat["priv_ids"]
        slots = torch.arange(PRIV_IDS, device=pid.device)
        pt = (self.emb_card(pid) + self.priv_slot(slots)[None]).mean(1) + self.priv_lin(b_flat["priv_f"])
        z = torch.cat([enc["h"].to(pt.dtype), self.priv_mlp(pt)], -1)
        logits = self.value(z).float()
        p = logits.softmax(-1)
        v = p[:, 0] - p[:, 2] + self.v_res(z).float().squeeze(-1)
        return logits, v

    def evaluate(self, b: dict, state=None, need_value: bool = True) -> dict:
        """Log-probabilities of the actions in ``b['lab']`` (teacher forcing), per-part terms
        for BC, entropies for PPO, and the value outputs. All outputs are [B*T]."""
        bsz, t = b["glob"].shape[:2]
        n = bsz * t
        enc, new_state = self.trunk(b, state)
        fl = lambda k: b[k].reshape(n, *b[k].shape[2:])  # noqa: E731
        lab = fl("lab").long()
        cand, ccost, elixir = fl("cand").bool(), fl("ccost").float(), fl("elixir").float()
        tm1, tm2 = fl("tm1").bool(), fl("tm2").bool()
        c1, t1, d1, c2, t2, d2 = (lab[:, i] for i in range(6))
        act = c1 >= 0
        can = cand.any(-1)
        ctx = self._ctx(enc)
        mem = enc["mem"]
        out = {}
        g = self.gate(ctx).float().squeeze(-1)
        out["gate_logit"] = g
        lp_gate = torch.where(act, Fn.logsigmoid(g), Fn.logsigmoid(-g))
        lp_gate = torch.where(can, lp_gate, torch.zeros_like(lp_gate))
        # first candidate
        lc1 = masked_log_softmax(self._cand_logits(self.q1(ctx), mem), cand)
        c1s = c1.clamp_min(0)
        lp_c1 = lc1.gather(1, c1s[:, None]).squeeze(1)
        mem1 = self._gather(mem, c1s)
        tl1, keys = self._tile_logits(ctx, mem1, enc["tiles"])
        lt1 = masked_log_softmax(tl1, tm1)
        has_t1 = act & (c1 < HAND) & (t1 >= 0)
        t1s = t1.clamp_min(0)
        lp_t1 = torch.where(has_t1, lt1.gather(1, t1s[:, None]).squeeze(1), torch.zeros_like(g))
        key1 = torch.where((act & (c1 < HAND) & (t1 >= 0))[:, None], self._gather(keys, t1s), torch.zeros_like(keys[:, 0]))
        ld1 = torch.log_softmax(self.delay(torch.cat([ctx, mem1, key1], -1)).float(), -1)
        d1s = d1.clamp(0, NDELAY - 1)
        lp_d1 = ld1.gather(1, d1s[:, None]).squeeze(1)
        # continue and second play (shadow state: elixir left after the first, first delay)
        emb1 = self._act_emb(mem1, key1, d1s)
        idx = torch.arange(NCAND, device=g.device)[None]
        left = elixir - self._gather(ccost, c1s)
        # exactly the support ``act`` samples from; collation already dropped labels outside it
        m2 = cand & (idx != c1s[:, None]) & (ccost <= left[:, None] + SECOND_PLAY_SLACK)
        has_c2 = c2 >= 0
        can2 = act & m2.any(-1)
        k = self.cont(torch.cat([ctx, emb1], -1)).float().squeeze(-1)
        lp_k = torch.where(can2, torch.where(has_c2, Fn.logsigmoid(k), Fn.logsigmoid(-k)), torch.zeros_like(k))
        ctx2 = ctx + self.shadow(torch.cat([emb1, (left / 10.0)[:, None].to(emb1.dtype),
                                            (d1s.float() / 4.0)[:, None].to(emb1.dtype)], -1))
        lc2 = masked_log_softmax(self._cand_logits(self.q2(ctx2), mem), m2)
        c2s = c2.clamp_min(0)
        lp_c2 = lc2.gather(1, c2s[:, None]).squeeze(1)
        mem2 = self._gather(mem, c2s)
        tl2, _ = self._tile_logits(ctx2, mem2, enc["tiles"])
        lt2 = masked_log_softmax(tl2, tm2)
        has_t2 = has_c2 & (c2 < HAND) & (t2 >= 0)
        t2s = t2.clamp_min(0)
        lp_t2 = torch.where(has_t2, lt2.gather(1, t2s[:, None]).squeeze(1), torch.zeros_like(g))
        key2 = torch.where(has_t2[:, None], self._gather(keys, t2s), torch.zeros_like(keys[:, 0]))
        dmask = torch.arange(NDELAY, device=g.device)[None] >= d1s[:, None]
        ld2 = masked_log_softmax(self.delay(torch.cat([ctx2, mem2, key2], -1)), dmask)
        d2s = torch.maximum(d2.clamp(0, NDELAY - 1), d1s)
        lp_d2 = ld2.gather(1, d2s[:, None]).squeeze(1)
        z = torch.zeros_like(g)
        lp_c1 = torch.where(act, lp_c1, z)
        lp_d1 = torch.where(act, lp_d1, z)
        lp_c2, lp_d2 = torch.where(has_c2, lp_c2, z), torch.where(has_c2, lp_d2, z)
        out.update(lp_gate=lp_gate, lp_c1=lp_c1, lp_t1=lp_t1, lp_d1=lp_d1, lp_k=lp_k, lp_c2=lp_c2,
                   lp_t2=lp_t2, lp_d2=lp_d2, can=can, act=act, has_t1=has_t1, can2=can2,
                   has_c2=has_c2, has_t2=has_t2)
        out["logp"] = lp_gate + lp_c1 + lp_t1 + lp_d1 + lp_k + lp_c2 + lp_t2 + lp_d2
        # entropies along the taken path (FirstLight: the parts that were chosen)
        pg = torch.sigmoid(g)
        h_gate = torch.where(can, Fn.binary_cross_entropy_with_logits(g, pg, reduction="none"), z)
        h1 = masked_entropy(lc1, cand) + torch.where(has_t1, masked_entropy(lt1, tm1), z) + masked_entropy(ld1, torch.ones_like(ld1, dtype=torch.bool))
        pk = torch.sigmoid(k)
        hk = torch.where(can2, Fn.binary_cross_entropy_with_logits(k, pk, reduction="none"), z)
        h2 = masked_entropy(lc2, m2) + torch.where(has_t2, masked_entropy(lt2, tm2), z) + masked_entropy(ld2, dmask)
        out["entropy"] = h_gate + torch.where(act, h1 + hk, z) + torch.where(has_c2, h2, z)
        if need_value:
            fb = {"priv_ids": fl("priv_ids").long(), "priv_f": fl("priv_f").float()}
            out["v_logits"], out["v"] = self.value_out(enc, fb)
        out["state"] = new_state
        return out

    def forward(self, b: dict, state=None, need_value: bool = True) -> dict:
        return self.evaluate(b, state, need_value)

    @torch.no_grad()
    def act(self, b: dict, state, card_temp: float = 1.0, gate_temp: float = 1.0, greedy: bool = False,
            gate_greedy: bool | None = None):
        """One decision for N seats. ``b`` has [N, ...] tensors (no T) and ``tiles`` [N, 4, 576].
        ``greedy`` takes the argmax of card, tile, delay and continue; the gate follows
        ``gate_greedy`` (default: ``greedy``). An argmax gate almost never acts at 250 ms
        decisions, so evaluation samples it. Returns lab [N, 6], logp [N], value [N],
        value logits, new state. ``logp`` is under the untempered policy."""
        gate_greedy = greedy if gate_greedy is None else gate_greedy
        bt = {k: v[:, None] for k, v in b.items() if k not in ("tiles",)}
        enc, new_state = self.trunk(bt, state)
        n = b["glob"].shape[0]
        dev = b["glob"].device
        cand, ccost = b["cand"].bool(), b["ccost"].float()
        # exact elixir, stored with every rollout step, so ``evaluate`` rebuilds the same
        # second-play mask (and the same log-prob)
        elixir = b["elixir"].float()
        tiles_all = b["tiles"].bool()
        ctx = self._ctx(enc)
        mem = enc["mem"]
        z = torch.zeros(n, device=dev)
        can = cand.any(-1)

        def pick(logp, mask, temp):
            if greedy:
                return logp.masked_fill(~mask, NEG).argmax(-1)
            lg = (logp / max(temp, 1e-3)).masked_fill(~mask, NEG)
            return torch.distributions.Categorical(logits=lg).sample()

        g = self.gate(ctx).float().squeeze(-1)
        if gate_greedy:
            act = g > 0
        else:
            act = torch.rand(n, device=dev) < torch.sigmoid(g / gate_temp)
        act = act & can
        lc1 = masked_log_softmax(self._cand_logits(self.q1(ctx), mem), cand)
        c1 = pick(lc1, cand, card_temp)
        mem1 = self._gather(mem, c1)
        tl1, keys = self._tile_logits(ctx, mem1, enc["tiles"])
        is_card1 = c1 < HAND
        tm1 = tiles_all[torch.arange(n, device=dev), c1.clamp(max=HAND - 1)] & is_card1[:, None]
        lt1 = masked_log_softmax(tl1, tm1)
        t1 = torch.where(is_card1, pick(lt1, tm1 | ~tm1.any(-1, keepdim=True), card_temp), torch.full_like(c1, -1))
        key1 = torch.where((t1 >= 0)[:, None], self._gather(keys, t1.clamp_min(0)), torch.zeros_like(keys[:, 0]))
        ld1 = torch.log_softmax(self.delay(torch.cat([ctx, mem1, key1], -1)).float(), -1)
        d1 = pick(ld1, torch.ones_like(ld1, dtype=torch.bool), 1.0)
        # a card with no legal tile cannot be played: fall back to waiting (decided before the
        # continue step, so the log-prob below covers exactly what ``evaluate`` will score)
        act = act & ~(is_card1 & ~tm1.any(-1))
        emb1 = self._act_emb(mem1, key1, d1)
        idx = torch.arange(NCAND, device=dev)[None]
        left = elixir - self._gather(ccost, c1)
        m2 = cand & (idx != c1[:, None]) & (ccost <= left[:, None] + SECOND_PLAY_SLACK)
        can2 = act & m2.any(-1)
        k = self.cont(torch.cat([ctx, emb1], -1)).float().squeeze(-1)
        more = (k > 0) if greedy else (torch.rand(n, device=dev) < torch.sigmoid(k))
        more = more & can2
        ctx2 = ctx + self.shadow(torch.cat([emb1, (left / 10.0)[:, None].to(emb1.dtype),
                                            (d1.float() / 4.0)[:, None].to(emb1.dtype)], -1))
        lc2 = masked_log_softmax(self._cand_logits(self.q2(ctx2), mem), m2)
        c2 = pick(lc2, m2 | ~m2.any(-1, keepdim=True), card_temp)
        mem2 = self._gather(mem, c2)
        tl2, _ = self._tile_logits(ctx2, mem2, enc["tiles"])
        is_card2 = c2 < HAND
        tm2 = tiles_all[torch.arange(n, device=dev), c2.clamp(max=HAND - 1)] & is_card2[:, None]
        lt2 = masked_log_softmax(tl2, tm2)
        t2 = torch.where(is_card2, pick(lt2, tm2 | ~tm2.any(-1, keepdim=True), card_temp), torch.full_like(c2, -1))
        key2 = torch.where((t2 >= 0)[:, None], self._gather(keys, t2.clamp_min(0)), torch.zeros_like(keys[:, 0]))
        dmask = torch.arange(NDELAY, device=dev)[None] >= d1[:, None]
        ld2 = masked_log_softmax(self.delay(torch.cat([ctx2, mem2, key2], -1)), dmask)
        d2 = pick(ld2, dmask, 1.0)
        more = more & ~(is_card2 & ~tm2.any(-1))
        neg = torch.full_like(c1, -1)
        lab = torch.stack([
            torch.where(act, c1, neg), torch.where(act, t1, neg), torch.where(act, d1, neg),
            torch.where(more, c2, neg), torch.where(more, t2, neg), torch.where(more, d2, neg)], -1)
        # exact log-prob of the sampled action under the untempered policy
        lp = torch.where(can, torch.where(act, Fn.logsigmoid(g), Fn.logsigmoid(-g)), z)
        lp = lp + torch.where(act, lc1.gather(1, c1[:, None]).squeeze(1) + ld1.gather(1, d1[:, None]).squeeze(1), z)
        lp = lp + torch.where(act & (t1 >= 0), lt1.gather(1, t1.clamp_min(0)[:, None]).squeeze(1), z)
        lp = lp + torch.where(can2, torch.where(more, Fn.logsigmoid(k), Fn.logsigmoid(-k)), z)
        lp = lp + torch.where(more, lc2.gather(1, c2[:, None]).squeeze(1) + ld2.gather(1, d2[:, None]).squeeze(1), z)
        lp = lp + torch.where(more & (t2 >= 0), lt2.gather(1, t2.clamp_min(0)[:, None]).squeeze(1), z)
        fb = {"priv_ids": b["priv_ids"].long(), "priv_f": b["priv_f"].float()}
        v_logits, v = self.value_out(enc, fb)
        return lab, lp, v, v_logits, new_state


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_model() -> AtlasNet:
    return AtlasNet()


# evaluation decodings (Stage 5): gate always sampled at T=1
DECODES = {
    "sample": dict(card_temp=1.0, greedy=False),
    "t05": dict(card_temp=0.5, greedy=False),
    "argmax": dict(greedy=True, gate_greedy=False),
}


def load_policy(path, device) -> AtlasNet:
    """A checkpoint (training dict with ``model``, or a bare / float16 state dict) -> eval model."""
    from .common import torch_load

    ck = torch_load(path)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = build_model()
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    return m.to(device).eval()


def to_device_batch(b: dict, device, non_blocking: bool = True) -> dict:
    out = {}
    for k, v in b.items():
        t = torch.as_tensor(v)
        if t.dtype == torch.float16 or t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device, non_blocking=non_blocking)
    return out
