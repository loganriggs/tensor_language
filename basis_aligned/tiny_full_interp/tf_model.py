"""The tiny bilinear transformer -- parameterized by ARCHITECTURE VARIANT as well
as width/depth/heads -- plus the exact fold machinery (rung 1-2).

Every mechanism here is transcribed from the parent program ../qk_mdl so the fold
identities and the cross-program comparisons hold:
  vanilla   qk_tokenline_train.MiniBilin (variant A)
  slots     qk_v8_train.V8Route + qk_e1_slotnorm_run.E1Route (per-slot RMSNorm)
  bandwidth qk_e15_reinvest_run.E15cRoute + solve_slot_c
  predicate qk_e22_predbasis_run.E22Route + match_kernels
  codebook  qk_e20_codebook_run / qk_e20b_vark_run (matching-pursuit, EMA)
  shrink    qk_e16_shrinkemb_run.E16Route + rem_schedule

FAMILY (unchanged across variants -- this is what makes the fold work)
  stream start   e = rms_norm(wte[idx])                 (GLOBAL RMSNorm)
  block entry    per-consumer token remnant + unit-weight sum of prior writes
                 (full visibility; the parent's dilation mask is meaningless at
                 depth 1-2).  For every variant except `shrink` the remnant is
                 just e at every consumer, i.e. the ordinary residual stream.
  attention      per-head RMSNorm on q/k THEN ROTARY.  POSITIONAL HANDLING IS
                 ROTARY, NOT LEARNED -- confirmed in ../qk_mdl/qk_tokenline_train
                 (rope_tables_exact / apply_rot), used unchanged by the whole
                 E-series through qk_e_common.py.
  pattern        pat = (q1.k1/hd) * (q2.k2/hd), causal-masked, NO SOFTMAX,
                 UNNORMALIZED (note: divided by hd, not sqrt(hd)); no value-lerp
  MLP            bilinear ungated  Down(Left(xn) * Right(xn)) + Down_bias
  readout        GLOBAL rms_norm, TIED embedding, logits = 30*tanh(./30)
  head_dim fixed 16, heads = Dc / 16, hidden = 4 * Dc

THREE WIDTHS (the E15c decoupling; equal for A/B/F)
  Dc  compute width   = cfg.width, the grid axis; attention is Dc-wide
  Ws  stream width    = Dc for A/B/F; = n_slots * s for C/D/E, where s is SOLVED
                        from live parameter counts so the body matches vanilla
  hidden = 4 * Dc

WHAT ROTARY MEANS FOR THE V x V TABLE (the program's core object).
Rotary makes the layer-0 score depend on the RELATIVE position only:
    s1_h(t at i, u at j) = q_h(t)^T R(i-j) k_h(u) / hd
so the exact layer-0 table is a V x V matrix PER RELATIVE DISTANCE delta, and it
is EXACTLY rank <= head_dim = 16 at every delta.  fold_layer0_qk() therefore
returns the (V, 16) factors -- the honest compressed form -- and materializes the
V x V matrix for any requested delta on demand.  Quoting "the V x V table"
without a delta would be wrong, so this code never does.

PREDICATE FOLD CAVEAT (recorded, not worked around): MATCH_prev[i,j] =
1[tok_{j-1} == tok_i] is a function of the query token and the token BEFORE the
key, so the predicate variant's pattern is exactly enumerable but over a
different index pair than (query token, key token).  fold_layer0_qk reports the
predicate parameters separately with that index set stated.

EXACT-REDUCTION GATE -- every variant with its distinguishing mechanism disabled
must reduce to its parent (variant_reduction_test):
  slots(n_slots=1, lasso 0, zero writes)   == vanilla
  bandwidth(s = Ws/n_slots, slot rows)     == slots
  predicate(terms zeroed / pred_on False)  == bandwidth   (bit-exact, 0.0)
  codebook(qz_on False)                    == bandwidth   (bit-exact, 0.0)
  shrink(mode 'control')                   == slots
"""
import copy
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_DIM = 16
EXP = 4
READ_NAMES = ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'Left', 'Right')
VARIANTS = ('vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink')
PARENT_OF = {'slots': 'vanilla', 'bandwidth': 'slots', 'predicate': 'bandwidth',
             'codebook': 'bandwidth', 'shrink': 'slots'}
QZ_DECAY, QZ_BETA, QZ_DEAD = 0.99, 0.25, 200        # parent's E20 constants


# ---------------------------------------------------------------- exact math
class exact_math:
    """tf32 OFF (and highest matmul precision) for the duration, applied
    SYMMETRICALLY around BOTH sides of every fold identity check.  The parent
    program lost a day to an asymmetric tf32 comparison (6e-4 of rounding read
    as 6e-4 of architecture); never compare a tf32 forward against an fp32
    reconstruction."""

    def __enter__(self):
        self.m = torch.backends.cuda.matmul.allow_tf32
        self.c = torch.backends.cudnn.allow_tf32
        self.p = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        return self

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.m
        torch.backends.cudnn.allow_tf32 = self.c
        torch.set_float32_matmul_precision(self.p)
        return False


# ---------------------------------------------------------------- config
@dataclass
class TFConfig:
    depth: int = 1
    width: int = 64                 # COMPUTE width Dc (the grid axis)
    vocab: int = 8192
    tok: str = 'bpe'                # 'bpe' = trained byte-level BPE (PRIMARY,
                                    # zero UNK); 'trunc' = top-K GPT-2 truncation
                                    # (comparison arm, 13-20% UNK)
    seed: int = 0
    variant: str = 'vanilla'
    head_dim: int = HEAD_DIM
    exp: int = EXP
    T: int = 512
    # --- B slots ---
    n_slots: int = 0                # 0 -> 2*depth (one per module write); 1 = off
    group_coeff: float = 3e-5       # in-loss group lasso (parent's E9a/E15c value)
    write_init: bool = False
    # --- C bandwidth ---
    small_dec: bool = False         # true-small decoders + solved stream width
    slot: int = 0                   # s; 0 -> solved by solve_slot()
    # --- D predicate ---
    predicate: bool = False
    # --- E codebook ---
    codebook: bool = False
    cb_n: int = 256
    cb_k_attn: int = 4
    cb_k_mlp: int = 2
    # --- F shrink ---
    shrink_mode: str = ''           # '' | 'shrink' | 'floor' | 'control'
    shrink_floor_slots: int = 1

    @property
    def n_heads(self):
        return self.width // self.head_dim

    @property
    def hidden(self):
        return self.exp * self.width

    @property
    def stream_width(self):
        return self.n_slots * self.slot

    def __post_init__(self):
        assert self.variant in VARIANTS, self.variant
        assert self.tok in ('bpe', 'trunc'), self.tok
        assert self.width % self.head_dim == 0, 'width must be a multiple of 16'
        v = self.variant
        if v == 'vanilla':
            self.n_slots, self.group_coeff = 1, 0.0
        else:
            self.write_init = True
            if self.n_slots == 0:
                self.n_slots = 2 * self.depth
        if v in ('bandwidth', 'predicate', 'codebook'):
            self.small_dec = True
        if v == 'predicate':
            self.predicate = True
        if v == 'codebook':
            self.codebook = True
        if v == 'shrink' and not self.shrink_mode:
            self.shrink_mode = 'floor'
        if not self.small_dec:
            # stream width == compute width; slots partition it
            assert self.width % self.n_slots == 0, \
                f'width {self.width} not divisible by n_slots {self.n_slots}'
            self.slot = self.width // self.n_slots

    def stem(self):
        # 'b' = trained byte-level BPE, 'v' = truncated GPT-2.  The tokenizer is
        # IN THE STEM so the two arms can never be silently compared or
        # overwritten by each other.
        v = f'b{self.vocab}' if self.tok == 'bpe' else f'v{self.vocab}'
        return (f'tf_{self.variant}_d{self.depth}_w{self.width}'
                f'_{v}_s{self.seed}')


def vanilla_body_params(cfg):
    """Body (non-embedding) parameter count of the VANILLA model at this
    depth/width, computed from the shapes -- the reinvestment budget."""
    Dc, H = cfg.width, cfg.hidden
    return cfg.depth * (5 * Dc * Dc + Dc * Dc + 2 * H * Dc + H * Dc + Dc)


def solve_slot(cfg):
    """C bandwidth: slot size s at fixed hidden/compute width such that the body
    matches vanilla.  DEPTH-PARAMETERIZED transcription of ../qk_mdl's
    solve_slot_c -- the literal 24 there is 2*DEPTH == n_slots and MUST become
    n_slots here.  Per block, params are exactly linear in s because the reads
    are Ws = n_slots*s wide:
        5 attn reads (Dc x Ws)      -> n_slots * 5 * Dc   per s
        Left, Right  (hidden x Ws)  -> n_slots * 2 * hidden per s
        c_proj (s x Dc)             -> Dc
        Down   (s x hidden)         -> hidden
        Down_bias (s,)              -> 1
    Returns (s, target, per_s)."""
    Dc, H, G = cfg.width, cfg.hidden, cfg.n_slots
    target = vanilla_body_params(cfg)
    per_s = G * (5 * Dc + 2 * H) + Dc + H + 1
    return max(1, int(round(target / (cfg.depth * per_s)))), target, per_s


# ---------------------------------------------------------------- rotary
def rope_tables_exact(T, hd, device='cpu', dtype=torch.float32):
    """Verbatim from ../qk_mdl/qk_tokenline_train.py (dtype added so the whole
    model, buffers included, can be rebuilt in fp64 for exactness checks)."""
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    t = torch.arange(T, dtype=dtype)
    fr = torch.outer(t, inv)
    return fr.cos().to(device), fr.sin().to(device)


def apply_rot(x, c, s):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)


def rot_matrix(delta, hd, device='cpu', dtype=torch.float32):
    """(hd, hd) matrix R with  q^T R(delta) k == <apply_rot(q, c_i, s_i),
    apply_rot(k, c_j, s_j)>  for delta = i - j.  Built by pushing the identity
    through the SAME apply_rot code the forward uses, so it cannot drift.

    `inv` is built at the SAME dtype rope_tables_exact uses, so the folded
    rotation and the forward's rotation round identically; building it in fp64
    and rounding to fp32 (what this did before) put a 1-ulp wedge between the
    two paths that showed up in the attention-table identity."""
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    th = float(delta) * inv
    c = th.cos().to(device)
    s = th.sin().to(device)
    return apply_rot(torch.eye(hd, device=device, dtype=dtype), c, s)


def match_kernels(idx, maskf):
    """Verbatim from ../qk_mdl/qk_e22_predbasis_run.py.
    MATCH_prev[b,i,j] = 1[tok_{j-1} == tok_i]  (prevtok[0] = -1)
    MATCH_same[b,i,j] = 1[tok_j == tok_i]      (diagonal included)
    dim 2 = query position i, dim 3 = key position j; causal-masked."""
    prevtok = torch.roll(idx, 1, 1)
    prevtok[:, 0] = -1
    qtok = idx.unsqueeze(2)
    kprev = (prevtok.unsqueeze(1) == qtok).float() * maskf
    ksame = (idx.unsqueeze(1) == qtok).float() * maskf
    return kprev, ksame


def mp_quantize(v, Cb, ksteps):
    """k-step matching pursuit against a unit-norm codebook (verbatim from
    ../qk_mdl/qk_e20_codebook_run.py).  v: (N, s); Cb: (n, s) unit rows."""
    r = v
    recon = torch.zeros_like(v)
    idxs, alphas, resids = [], [], []
    for _ in range(ksteps):
        ip = r @ Cb.t()
        i = ip.argmax(1)
        a = ip.gather(1, i[:, None]).squeeze(1)
        resids.append(r)
        idxs.append(i)
        alphas.append(a)
        recon = recon + a[:, None] * Cb[i]
        r = r - a[:, None] * Cb[i]
    return recon, idxs, alphas, resids


@torch.no_grad()
def ema_update(Cb, M, last_used, usage, step_now, x, idxs, alphas, resids,
               decay=QZ_DECAY, dead_t=QZ_DEAD):
    """EMA codebook update + dead-code reinit (verbatim from the parent)."""
    n = Cb.shape[0]
    acc = torch.zeros_like(M)
    counts = torch.zeros(n, dtype=torch.long, device=Cb.device)
    for i, a, r in zip(idxs, alphas, resids):
        acc.index_add_(0, i, a[:, None] * r)
        counts += torch.bincount(i, minlength=n)
    M.mul_(decay).add_(acc, alpha=1 - decay)
    nm = M.norm(dim=1)
    upd = nm > 1e-8
    Cb[upd] = M[upd] / nm[upd, None]
    last_used[counts > 0] = step_now
    usage += counts
    dead = (step_now - last_used) > dead_t
    if bool(dead.any()):
        ridx = torch.randint(0, x.shape[0], (int(dead.sum()),), device=x.device)
        cand = x[ridx]
        Cb[dead] = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-8)
        M[dead] = Cb[dead]
        last_used[dead] = step_now
    return counts


def rem_schedule(Ws, S, floor_slots, depth):
    """F shrink, verbatim from ../qk_mdl/qk_e16_shrinkemb_run.py: remnant width
    per consumer li = 0..depth (depth = the readout row).  li=0 is the full
    stream (identity, no matrix); each block reclaims 2 slots."""
    floor = floor_slots * S
    return [Ws] + [max(Ws - 2 * S * li, floor) for li in range(1, depth + 1)]


# ---------------------------------------------------------------- block
class TFBlock(nn.Module):
    def __init__(self, cfg, Ws):
        super().__init__()
        Dc, H, s = cfg.width, cfg.hidden, cfg.slot
        for nm in ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v'):
            setattr(self, nm, nn.Linear(Ws, Dc, bias=False))
        self.Left = nn.Linear(Ws, H, bias=False)
        self.Right = nn.Linear(Ws, H, bias=False)
        out = s if cfg.small_dec else Ws
        self.c_proj = nn.Linear(Dc, out, bias=False)
        self.Down = nn.Linear(H, out, bias=False)
        self.Down_bias = nn.Parameter(torch.zeros(out))


# ---------------------------------------------------------------- model
class TinyBilin(nn.Module):
    def __init__(self, cfg: TFConfig):
        super().__init__()
        self.cfg = cfg
        Ws = cfg.stream_width
        self.Ws, self.Dc, self.s = Ws, cfg.width, cfg.slot
        torch.manual_seed(cfg.seed)
        self.wte = nn.Embedding(cfg.vocab, Ws)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([TFBlock(cfg, Ws) for _ in range(cfg.depth)])
        # Write init: zero (vanilla, the parent's variant-A convention) or
        # nonzero 0.02/sqrt(2*depth) (the parent's dead-gradient escape),
        # fan-in rescaled off width 384 per AGENT_BRIEF.  Separate generator
        # 888 so the main RNG stream is untouched -- that is what makes the
        # variant reductions bit-exact.
        std = 0.02 / math.sqrt(2 * cfg.depth) * math.sqrt(384.0 / cfg.width)
        self.write_init_std = std if cfg.write_init else 0.0
        gw = torch.Generator().manual_seed(888)
        with torch.no_grad():
            for blk in self.h:
                for p in (blk.c_proj.weight, blk.Down.weight):
                    if cfg.write_init:
                        p.copy_(torch.randn(p.shape, generator=gw) * std)
                    else:
                        p.zero_()
        # write masks (full decoders only; small decoders scatter instead)
        wm = torch.zeros(2 * cfg.depth, Ws)
        for k in range(2 * cfg.depth):
            if cfg.n_slots <= 1:
                wm[k] = 1.0
            else:
                wm[k, cfg.slot * k:cfg.slot * (k + 1)] = 1.0
        self.register_buffer('wmask', wm)
        cos, sin = rope_tables_exact(cfg.T, cfg.head_dim, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask',
                             torch.tril(torch.ones(cfg.T, cfg.T, dtype=torch.bool)))
        # --- D predicate: zeros consume no RNG -> bit-exact with the parent ---
        self.pred_on = cfg.predicate
        if cfg.predicate:
            H = cfg.n_heads
            self.pred_prof = nn.Parameter(torch.zeros(cfg.depth, H, cfg.T))
            self.pred_b = nn.Parameter(torch.zeros(cfg.depth, H))
            self.pred_c = nn.Parameter(torch.zeros(cfg.depth, H))
            ar = torch.arange(cfg.T)
            self.register_buffer('offmat', (ar[:, None] - ar[None, :]).clamp(min=0))
        # --- E codebook: BUFFERS, separate generator 4242 (no RNG consumed) ---
        self.qz_on = cfg.codebook
        self._aux = None
        self._commit_sum, self._commit_n = None, 0
        if cfg.codebook:
            self.qz_ksteps = [cfg.cb_k_attn if k % 2 == 0 else cfg.cb_k_mlp
                              for k in range(2 * cfg.depth)]
            gq = torch.Generator().manual_seed(4242)
            cb = torch.randn(2 * cfg.depth, cfg.cb_n, cfg.slot, generator=gq)
            cb = cb / cb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.register_buffer('qz_codebook', cb.float())
            self.register_buffer('qz_ema', cb.float().clone())
            self.register_buffer('qz_last_used',
                                 torch.zeros(2 * cfg.depth, cfg.cb_n,
                                             dtype=torch.long))
            self.register_buffer('qz_usage',
                                 torch.zeros(2 * cfg.depth, cfg.cb_n,
                                             dtype=torch.long))
            self.register_buffer('qz_step', torch.zeros((), dtype=torch.long))
        # --- F shrink: per-consumer token remnants, identity-slice init ---
        self.shrink_mode = cfg.shrink_mode
        self.rem_dims = [Ws] * (cfg.depth + 1)
        self.W_rem = None
        if cfg.shrink_mode in ('shrink', 'floor'):
            fs = cfg.shrink_floor_slots if cfg.shrink_mode == 'floor' else 0
            self.rem_dims = rem_schedule(Ws, cfg.slot, fs, cfg.depth)
            self.W_rem = nn.ModuleDict()
            with torch.no_grad():
                for li in range(1, cfg.depth + 1):
                    r = self.rem_dims[li]
                    if r == 0:
                        continue
                    lin = nn.Linear(Ws, r, bias=False)
                    w = torch.zeros(r, Ws)
                    w[:, Ws - r:] = torch.eye(r)
                    lin.weight.copy_(w)
                    self.W_rem[str(li)] = lin
        # predicate params (3D / scalar mixtures) must NOT go to Muon
        self.muon_exclude = ('pred_',)

    # ---------------- pieces ----------------
    def slot_norm(self, x):
        """Per-slot RMSNorm at the module inputs.  n_slots == 1 reduces EXACTLY
        to the global RMSNorm (the identity-control mode)."""
        G = self.cfg.n_slots
        Dm = x.shape[-1]
        if G <= 1:
            return F.rms_norm(x, (Dm,))
        S = Dm // G
        sh = x.shape
        return F.rms_norm(x.view(*sh[:-1], G, S), (S,)).view(sh)

    def remnants(self, e):
        """Per-consumer token channel.  Non-shrink variants return e at every
        consumer, i.e. exactly the ordinary residual stream."""
        if self.W_rem is None:
            return [e] * (self.cfg.depth + 1)
        Ws = e.shape[-1]
        out = [e]
        for li in range(1, self.cfg.depth + 1):
            r = self.rem_dims[li]
            if r == 0:
                out.append(None)
                continue
            Rt = torch.zeros(*e.shape[:-1], Ws, device=e.device, dtype=e.dtype)
            Rt[..., Ws - r:] = self.W_rem[str(li)](e)
            out.append(Rt)
        return out

    def pred_terms(self, l, Kprev, Ksame, maskf, Tq):
        """(B, NH, Tq, Tq) named pattern terms for block l (verbatim E22)."""
        prof = self.pred_prof[l][:, self.offmat[:Tq, :Tq]] * maskf
        return (prof[None]
                + self.pred_b[l].view(1, -1, 1, 1) * Kprev[:, None]
                + self.pred_c[l].view(1, -1, 1, 1) * Ksame[:, None])

    def _qz_slot(self, k, v, cache, collect):
        ks = self.qz_ksteps[k]
        upd = self.training and torch.is_grad_enabled() and cache is not None
        with torch.autocast('cpu' if v.device.type == 'cpu' else 'cuda',
                            enabled=False):
            x32 = v.detach().float().reshape(-1, self.s)
            recon, idxs, alphas, resids = mp_quantize(x32, self.qz_codebook[k], ks)
            if upd:
                ema_update(self.qz_codebook[k], self.qz_ema[k],
                           self.qz_last_used[k], self.qz_usage[k],
                           int(self.qz_step), x32, idxs, alphas, resids)
            q = recon.view(v.shape)
        out = v + (q.to(v.dtype) - v).detach()          # straight-through
        if upd:
            commit = (v.float() - q.detach()).pow(2).mean()
            self._commit_sum = commit if self._commit_sum is None \
                else self._commit_sum + commit
            self._commit_n += 1
        if collect is not None:
            collect.setdefault('codes', {}).setdefault(k, []).append(
                torch.stack(idxs, 1).to(torch.int32).cpu())
        if cache is not None:
            cache[k] = out
        return out

    def _qz_full(self, full, n_written, cache, collect):
        """Quantize the post-slot-norm content of the already-written slots.
        qz_on False returns `full` unchanged (bit-exact bypass).  NOTE at depth
        1 only slot 0 is ever quantized (n_written is 0 at the attention input
        and 1 at the MLP input, and the last slot only reaches the readout,
        which the parent exempts) -- an honest structural consequence of the
        depth, recorded rather than patched."""
        if not self.qz_on or n_written == 0:
            return full
        s = self.s
        segs = []
        for k in range(2 * self.cfg.depth):
            v = full[..., s * k:s * (k + 1)]
            if k >= n_written:
                segs.append(v)
            elif cache is not None and k in cache:
                segs.append(cache[k])
            else:
                segs.append(self._qz_slot(k, v, cache, collect))
        return torch.cat(segs, dim=-1)

    def pop_aux_loss(self):
        a, self._aux = self._aux, None
        return a

    def write_out(self, w_small, k, B, Tq):
        """Place a module write into slot k: scatter (small decoder) or mask."""
        if self.cfg.small_dec:
            out = torch.zeros(B, Tq, self.Ws, device=w_small.device,
                              dtype=w_small.dtype)
            out[..., self.s * k:self.s * (k + 1)] = w_small
            return out
        if self.cfg.n_slots > 1:
            return w_small * self.wmask[k].to(w_small.dtype)
        return w_small

    # ---------------- forward ----------------
    def forward(self, idx, collect=None, pat_out=None):
        cfg = self.cfg
        Ws, Dc, H, hd = self.Ws, self.Dc, cfg.n_heads, cfg.head_dim
        B, Tq = idx.shape
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        maskf = mask.float()
        if self.pred_on:
            Kprev, Ksame = match_kernels(idx, maskf)
        if self.training and torch.is_grad_enabled() and self.qz_on:
            self.qz_step += 1
        self._commit_sum, self._commit_n = None, 0
        cache = {} if self.qz_on else None
        e = F.rms_norm(self.wte(idx), (Ws,))
        rem = self.remnants(e)
        streams = [None]

        def entry(li):
            tot = rem[li]
            for i in range(1, 1 + 2 * min(li, cfg.depth)):
                tot = streams[i] if tot is None else tot + streams[i]
            return tot

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect.setdefault('entry_norm', []).append(
                    x.detach().float().norm(dim=-1).mean().item())
            hn = self._qz_full(self.slot_norm(x), 2 * l, cache, collect)

            def qk(lin):
                z = lin(hn).view(B, Tq, H, hd)
                return apply_rot(F.rms_norm(z, (hd,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, H, hd)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if self.pred_on:
                pat = pat + self.pred_terms(l, Kprev, Ksame, maskf, Tq).to(pat.dtype)
            if pat_out is not None:
                pat_out.append(pat.detach())
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dc)
            aw = self.write_out(blk.c_proj(y), 2 * l, B, Tq)
            x = x + aw
            xn = self._qz_full(self.slot_norm(x), 2 * l + 1, cache, collect)
            mw = self.write_out(blk.Down(blk.Left(xn) * blk.Right(xn))
                                + blk.Down_bias, 2 * l + 1, B, Tq)
            if collect is not None:
                collect.setdefault('attn_write', []).append(aw.detach())
                collect.setdefault('mlp_write', []).append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = F.rms_norm(entry(cfg.depth), (Ws,))
        logits = x @ self.wte.weight.t()
        if self._commit_sum is not None:
            self._aux = QZ_BETA * self._commit_sum / self._commit_n
        return 30 * torch.tanh(logits / 30)

    # ---------------- group lasso ----------------
    def group_penalty(self):
        """Sum over blocks, read matrices and slot column groups of the group
        Frobenius norm ||M[:, s*k : s*(k+1)]||_F (the parent's group lasso)."""
        G, s = self.cfg.n_slots, self.s
        tot = None
        for blk in self.h:
            for nm in READ_NAMES:
                M = getattr(blk, nm).weight
                g = (M.pow(2).view(M.shape[0], G, s).sum(dim=(0, 2))
                     + 1e-12).sqrt().sum()
                tot = g if tot is None else tot + g
        return tot if tot is not None else \
            torch.zeros((), device=self.wte.weight.device)

    # ================================================================ FOLDS
    @torch.no_grad()
    def token_input_table(self, dtype=torch.float32):
        """E_hn: the EXACT layer-0 module input for every vocabulary item,
        (V, Ws).  A deterministic function of wte alone -- no data.  (Layer 0's
        entry is the full-width remnant for every variant, and no slot has been
        written yet, so the codebook never touches it.)"""
        e = F.rms_norm(self.wte.weight.detach().to(dtype), (self.Ws,))
        return self.slot_norm(e)

    @torch.no_grad()
    def fold_layer0_qk(self, deltas=(0,), materialize=True, heads=None,
                       device=None, dtype=torch.float32):
        """RUNG 1-2, ATTENTION.  Exact layer-0 token-pair score tables.

        Returns
          'Q1','K1','Q2','K2' : (H, V, hd) factors, Q = rms_norm_hd(W_q Ehn)
          'Vv'                : (H, V, hd) the layer-0 value table
          'tables'            : {delta: {'s1': (H,V,V), 's2': (H,V,V)}} when
                                materialize; s1[h,t,u] is the EXACT branch-1
                                score for query token t at position i and key
                                token u at position j = i - delta.  The full
                                pattern is s1*s2 (+ predicate terms).
          'rank_bound'        : head_dim (each table is exactly rank <= 16)
          'predicate'         : {'prof': (H,T), 'b': (H,), 'c': (H,),
                                 'index_sets': ...} for the predicate variant.
        Materialization is O(H V^2) (67 MB per head-branch at V=4096) so it is
        done head-by-head and returned on CPU.  `dtype` exists because the
        table product q^T R k is CANCELLATION-HEAVY (16 terms of magnitude ~hd
        summing to an O(1) result): fp32 is right for comparing against an fp32
        forward, fp64 is required for a sharp known-answer check."""
        cfg = self.cfg
        H, hd, V = cfg.n_heads, cfg.head_dim, cfg.vocab
        dev = device or self.wte.weight.device
        Ehn = self.token_input_table(dtype).to(dev)
        blk = self.h[0]

        def proj(lin, norm=True):
            z = (Ehn @ lin.weight.detach().to(dtype).to(dev).t()).view(V, H, hd)
            if norm:
                z = F.rms_norm(z, (hd,))
            return z.permute(1, 0, 2).contiguous()

        out = {'Q1': proj(blk.c_q), 'K1': proj(blk.c_k), 'Q2': proj(blk.c_q2),
               'K2': proj(blk.c_k2), 'Vv': proj(blk.c_v, norm=False),
               'rank_bound': hd, 'deltas': list(deltas), 'vocab': V,
               'n_heads': H, 'head_dim': hd}
        if self.pred_on:
            out['predicate'] = {
                'prof': self.pred_prof[0].detach().float().cpu(),
                'b': self.pred_b[0].detach().float().cpu(),
                'c': self.pred_c[0].detach().float().cpu(),
                'index_sets': 'prof indexed by relative distance i-j; '
                              'b multiplies 1[tok_{j-1} == tok_i] -- a table '
                              'over (query token, token BEFORE the key), NOT '
                              '(query token, key token); c multiplies '
                              '1[tok_j == tok_i], the V x V identity.'}
        hs = list(range(H)) if heads is None else list(heads)
        if materialize:
            tabs = {}
            for d in deltas:
                R = rot_matrix(d, hd, dev, dtype)
                s1 = torch.empty(len(hs), V, V, dtype=dtype)
                s2 = torch.empty(len(hs), V, V, dtype=dtype)
                for a, h in enumerate(hs):
                    s1[a] = ((out['Q1'][h] @ R) @ out['K1'][h].t() / hd).cpu()
                    s2[a] = ((out['Q2'][h] @ R) @ out['K2'][h].t() / hd).cpu()
                tabs[int(d)] = {'s1': s1, 's2': s2, 'heads': hs}
            out['tables'] = tabs
        return out

    @torch.no_grad()
    def fold_mlp(self, li=0, device=None, chunk=None, dtype=None):
        """RUNG 1, MLP.  The exact symmetric third-order tensor
            T[o,i,j] = 1/2 sum_f Down[o,f] (Left[f,i] Right[f,j]
                                            + Left[f,j] Right[f,i])
        with  MLP(z) = T(z,z) + Down_bias  exactly, and (RMSNorm gauge)
            MLP(rms_norm(x)) = T(x,x) * Ws / ||x||^2 + Down_bias.
        Shape (out, Ws, Ws) -- `out` is s for small decoders, Ws otherwise."""
        blk = self.h[li]
        dev = device or self.wte.weight.device
        dt = dtype or torch.float32
        Dn = blk.Down.weight.detach().to(dt).to(dev)
        L = blk.Left.weight.detach().to(dt).to(dev)
        R = blk.Right.weight.detach().to(dt).to(dev)
        O, Ws = Dn.shape[0], L.shape[1]
        chunk = chunk or max(1, min(O, 1 + (1 << 24) // max(1, Ws * Ws)))
        T = torch.empty(O, Ws, Ws, dtype=dt)
        for a in range(0, O, chunk):
            b = min(a + chunk, O)
            M = torch.einsum('of,fi,fj->oij', Dn[a:b], L, R)
            T[a:b] = (0.5 * (M + M.transpose(1, 2))).cpu()
        return T

    @torch.no_grad()
    def fold_forward(self, idx, fold=None, mlp_tensors=None):
        """Recompute the logits using ONLY folded objects: the layer-0 token
        tables (Q/K/V factors + rotary), the decoder matrices, and the per-layer
        symmetric MLP tensors.  Deeper attention layers are NOT token-foldable
        (their input is not token-determined) and are computed from the weights
        -- that is exactly the honest statement of what folds.  Variant
        mechanisms (predicate terms, quantization, remnants) are reused rather
        than re-derived: the gate certifies the FOLD objects, not those helpers.
        Deliberately a separate implementation from forward().

        DTYPE: every object is built at the module's own parameter dtype (`dt`),
        so `model.double()` gives a genuine fp64 comparison instead of silently
        mixing an fp32 fold into an fp64 forward.  Nothing here is hard-cast to
        float32 any more."""
        cfg = self.cfg
        Ws, Dc, H, hd = self.Ws, self.Dc, cfg.n_heads, cfg.head_dim
        B, Tq = idx.shape
        dev = idx.device
        dt = self.wte.weight.dtype
        fold = fold or self.fold_layer0_qk(materialize=False, device=dev,
                                           dtype=dt)
        mlp_tensors = mlp_tensors or [
            self.fold_mlp(li, device=dev, dtype=dt).to(dev)
            for li in range(cfg.depth)]
        cos = self.cos[None, :Tq, None, :].to(dt)
        sin = self.sin[None, :Tq, None, :].to(dt)
        mask = self.mask[:Tq, :Tq]
        maskf = mask.to(dt)
        if self.pred_on:
            Kprev, Ksame = match_kernels(idx, maskf)
        cache = {} if self.qz_on else None
        e = F.rms_norm(self.wte(idx).to(dt), (Ws,))
        rem = self.remnants(e)
        streams = [None]

        def entry(li):
            tot = rem[li]
            for i in range(1, 1 + 2 * min(li, cfg.depth)):
                tot = streams[i] if tot is None else tot + streams[i]
            return tot

        for l, blk in enumerate(self.h):
            x = entry(l)
            hn = self._qz_full(self.slot_norm(x), 2 * l, cache, None)
            if l == 0:
                # token tables: index the (H,V,hd) factors by the token ids
                q = apply_rot(fold['Q1'].permute(1, 0, 2)[idx], cos, sin)
                k = apply_rot(fold['K1'].permute(1, 0, 2)[idx], cos, sin)
                q2 = apply_rot(fold['Q2'].permute(1, 0, 2)[idx], cos, sin)
                k2 = apply_rot(fold['K2'].permute(1, 0, 2)[idx], cos, sin)
                v = fold['Vv'].permute(1, 0, 2)[idx]
            else:
                def qkf(lin):
                    z = (hn @ lin.weight.detach().to(dt).t()).view(B, Tq, H, hd)
                    return apply_rot(F.rms_norm(z, (hd,)), cos, sin)

                q, k = qkf(blk.c_q), qkf(blk.c_k)
                q2, k2 = qkf(blk.c_q2), qkf(blk.c_k2)
                v = (hn @ blk.c_v.weight.detach().to(dt).t()).view(B, Tq, H, hd)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if self.pred_on:
                pat = pat + self.pred_terms(l, Kprev, Ksame, maskf, Tq)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dc)
            aw = self.write_out(y @ blk.c_proj.weight.detach().to(dt).t(),
                                2 * l, B, Tq)
            x = x + aw
            xn = self._qz_full(self.slot_norm(x), 2 * l + 1, cache, None)
            mw = self.write_out(
                torch.einsum('oij,bti,btj->bto', mlp_tensors[l], xn, xn)
                + blk.Down_bias.detach().to(dt), 2 * l + 1, B, Tq)
            streams.append(aw)
            streams.append(mw)
        x = F.rms_norm(entry(cfg.depth), (Ws,))
        return 30 * torch.tanh((x @ self.wte.weight.detach().to(dt).t()) / 30)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def body_params(self):
        return self.n_params() - self.wte.weight.numel()


# ---------------------------------------------------------------- factories
def make_model(cfg: TFConfig, device='cuda'):
    if cfg.small_dec and not cfg.slot:
        cfg.slot = solve_slot(cfg)[0]
    return TinyBilin(cfg).to(device)


def config_from_stem(stem):
    p = stem.split('_')
    kv = {x[0]: x[1:] for x in p[2:]}
    tok = 'bpe' if 'b' in kv else 'trunc'
    return TFConfig(variant=p[1], depth=int(kv['d']), width=int(kv['w']),
                    vocab=int(kv.get('b', kv.get('v'))), tok=tok,
                    seed=int(kv['s']))


# ================================================================ CONTROLS
def _maxdiff(a, b):
    return float((a - b).abs().max())


def cast_model(model, dtype=torch.float64):
    """DEEP COPY of `model` with parameters AND floating buffers at `dtype`,
    with the rotary tables REBUILT at that dtype.

    Rebuilding matters: `.double()` alone merely upcasts the fp32 cos/sin, so an
    'fp64' check would still carry fp32 rounding in the rotary and the rotary,
    not the fold, would set the error floor.  The bool causal mask and the
    integer codebook counters are left alone by .to(dtype)."""
    m = copy.deepcopy(model).to(dtype)
    dev = m.wte.weight.device
    cos, sin = rope_tables_exact(m.cfg.T, m.cfg.head_dim, 'cpu', dtype)
    m.cos, m.sin = cos.to(dev), sin.to(dev)
    return m.eval()


@torch.no_grad()
def _fold_identity_body(model, idx, verbose=False):
    """The three algebraic identities plus the end-to-end one, all evaluated at
    whatever dtype `model` already carries.  Returns (numbers, reference
    logits) -- no verdict, and the logits so the caller can calibrate the fp32
    noise floor of the reference itself against fp64."""
    res = {}
    cfg = model.cfg
    dev = idx.device
    Ws = model.Ws
    dt = model.wte.weight.dtype
    g = torch.Generator(device='cpu').manual_seed(20260808)
    for li in range(cfg.depth):
        T = model.fold_mlp(li, device=dev, dtype=dt).to(dev)
        blk = model.h[li]
        z = torch.randn(64, Ws, generator=g, dtype=dt).to(dev)
        direct = blk.Down(blk.Left(z) * blk.Right(z))
        folded = torch.einsum('oij,bi,bj->bo', T, z, z)
        # denominator floored: a ZERO-init decoder makes both sides exactly
        # 0 and 0/0 would read as NaN.  Such a check is vacuous, not
        # passing -- __main__ perturbs zero decoders before gating.
        res[f'mlp_tensor_identity_l{li}_relmax'] = float(
            (direct - folded).abs().max() / direct.abs().max().clamp_min(1e-30))
        x = torch.randn(64, Ws, generator=g, dtype=dt).to(dev) * 7.0
        zz = F.rms_norm(x, (Ws,))
        d2 = blk.Down(blk.Left(zz) * blk.Right(zz))
        gauge = torch.einsum('oij,bi,bj->bo', T, x, x) \
            * (Ws / x.pow(2).sum(-1, keepdim=True))
        res[f'mlp_rmsnorm_gauge_l{li}_relmax'] = float(
            (d2 - gauge).abs().max() / d2.abs().max().clamp_min(1e-30))
        res[f'decoder_absmax_l{li}'] = float(blk.Down.weight.abs().max())
        del T
    # ---- (c) layer-0 attention tables ----
    pats = []
    _ = model(idx, pat_out=pats)
    pat0 = pats[0].to(dt)
    B, H, Tq, _ = pat0.shape
    ds = tuple(d for d in (0, 1, 2, 5) if d < Tq)
    fold = model.fold_layer0_qk(deltas=ds, materialize=True, device=dev,
                                dtype=dt)
    worst = 0.0
    maskf = model.mask[:Tq, :Tq].to(dt)
    pt = model.pred_terms(0, *match_kernels(idx, maskf), maskf, Tq) \
        if model.pred_on else None
    for d in ds:
        i = torch.arange(d, Tq, device=dev)
        j = i - d
        s1 = fold['tables'][d]['s1'].to(dev)
        s2 = fold['tables'][d]['s2'].to(dev)
        qt, kt = idx[:, i], idx[:, j]
        rebuilt = torch.stack(
            [s1[h][qt, kt] * s2[h][qt, kt] for h in range(H)], 1)
        if pt is not None:
            rebuilt = rebuilt + pt[:, :, i, j]
        ref = pat0[:, :, i, j]
        worst = max(worst, float((rebuilt - ref).abs().max()
                                 / ref.abs().max().clamp_min(1e-12)))
        del s1, s2
    res['attn_layer0_table_identity_relmax'] = worst
    del fold
    # ---- (d) whole model ----
    ref = model(idx)
    rec = model.fold_forward(idx)
    res['fold_forward_max_logit_diff'] = _maxdiff(rec, ref)
    res['logit_absmax'] = float(ref.abs().max())
    res['fold_forward_rel_logit_diff'] = float(
        res['fold_forward_max_logit_diff']
        / max(res['logit_absmax'], 1e-30))
    if verbose:
        for k, v in res.items():
            print(f'    [{dt}] {k}: {v}', flush=True)
    return res, ref


@torch.no_grad()
def check_fold_identities(model, idx, tol_attn=1e-5, tol_mlp=1e-5,
                          tol_logit_rel=1e-5, tol_logit_fp64=1e-9,
                          tol_algebraic_fp64=1e-12,
                          tol_noise_multiple=10.0, fp64=True, verbose=True):
    """HARD GATE.  Both sides inside exact_math() (tf32 off SYMMETRICALLY).
      (a) MLP tensor identity   T(z,z) == Down(Left(z)*Right(z))
      (b) RMSNorm gauge         MLP(rms(x)) == T(x,x)*Ws/||x||^2
      (c) layer-0 attention table identity at several relative distances
      (d) whole-model identity  fold_forward == forward

    Criterion for (d), corrected 2026-08-08.  The old gate compared an ABSOLUTE
    max logit difference against 1e-5 while every other check was relative.
    That is not a scale-free criterion: these logits live on 30*tanh(.../30) so
    they reach ~10-20, and one fp32 ulp at 16 is already 1e-6 -- an absolute
    1e-5 budget is ~8 ulps for a forward pass that accumulates thousands of
    fp32 roundings.  Four of six trained cells failed it while EVERY algebraic
    identity passed at 2e-7, which is the signature of a precision floor, not a
    wrong fold.  The corrected gate has three parts and is strictly STRONGER:

      * fp64 ABSOLUTE  max|diff| < tol_logit_fp64 (1e-9).  THIS is the real
        exactness gate.  If the fold were algebraically wrong the fp64 residual
        would sit at the size of the error; observed values are ~3e-14, i.e.
        ~10 fp64 ulps at logit magnitude 15.
      * fp32 RELATIVE  max|diff| / max|logit| < tol_logit_rel (1e-5).  A
        scale-free version of the old check.  Justification: fp32 eps = 1.19e-7
        and the folded and unfolded paths perform the same O(1e3-1e4)
        multiply-accumulates in a DIFFERENT ORDER, so a random-walk rounding
        bound is ~sqrt(N)*eps ~ 1e-5 relative.
      * CALIBRATED against the reference's own fp32 noise.  The gate also
        measures max|forward_fp32 - forward_fp64| / max|logit| -- how far the
        UNFOLDED forward is from its own exact value -- and requires the
        fold-vs-forward disagreement to be no more than tol_noise_multiple
        (10x) that floor.  This is the part that would catch a genuine but
        small bug hiding under a fixed 1e-5: a correct fold cannot disagree
        with the forward by much more than the forward disagrees with itself.
    """
    res = {}
    with exact_math():
        r32, ref32 = _fold_identity_body(model.float().eval(), idx,
                                         verbose=False)
        res.update(r32)
        if fp64:
            m64 = cast_model(model, torch.float64)
            r64, ref64 = _fold_identity_body(m64, idx, verbose=False)
            for k, v in r64.items():
                res[f'fp64_{k}'] = v
            sc = max(float(ref64.abs().max()), 1e-30)
            res['fp32_forward_vs_fp64_forward_rel'] = float(
                (ref32.double() - ref64).abs().max()) / sc
            res['fold_vs_forward_over_fp32_selfnoise'] = float(
                res['fold_forward_rel_logit_diff']
                / max(res['fp32_forward_vs_fp64_forward_rel'], 1e-30))
            del m64
    fp32_ok = bool(
        all(v < tol_mlp for k, v in res.items() if k.startswith('mlp_'))
        and res['attn_layer0_table_identity_relmax'] < tol_attn
        and res['fold_forward_rel_logit_diff'] < tol_logit_rel)
    fp64_ok = (not fp64) or bool(
        all(v < tol_algebraic_fp64
            for k, v in res.items() if k.startswith('fp64_mlp_'))
        and res['fp64_attn_layer0_table_identity_relmax'] < tol_algebraic_fp64
        and res['fp64_fold_forward_max_logit_diff'] < tol_logit_fp64
        and res['fold_vs_forward_over_fp32_selfnoise'] < tol_noise_multiple)
    res['fp32_sanity_pass'] = fp32_ok
    res['fp64_exactness_pass'] = fp64_ok
    res['pass'] = bool(fp32_ok and fp64_ok)
    res['criterion'] = {
        'fp32_sanity': {'mlp_rel': tol_mlp, 'attn_rel': tol_attn,
                        'logit_rel': tol_logit_rel},
        'fp64_exactness': {'algebraic_rel': tol_algebraic_fp64,
                           'logit_abs': tol_logit_fp64,
                           'noise_multiple': tol_noise_multiple},
        'note': 'TWO-TIER.  fp32 tier is a scale-free sanity band sized by '
                'sqrt(N)*eps_fp32 ~ 1e-5; fp64 tier is the real exactness '
                'gate (algebraic identities to 1e-12 relative, end-to-end to '
                '1e-9 absolute) plus a calibration requiring the fold-vs-'
                'forward gap to stay within 10x the forward pass own fp32-vs-'
                'fp64 self-noise.  SUPERSEDED: the old absolute-1e-5-fp32 '
                'logit criterion and the 1e-6 fp32 algebraic criteria, which '
                'together failed 5/6 cells on rounding alone while every fp64 '
                'residual sat at 1e-14 to 1e-16.'}
    if verbose:
        for k, v in res.items():
            print(f'  fold gate {k}: {v}', flush=True)
    return res


@torch.no_grad()
def gate_negative_control(device='cuda', width=32, vocab=128, depth=1):
    """NEGATIVE CONTROL FOR THE GATE ITSELF.  A gate that has just been LOOSENED
    is worthless unless it still fails on a wrong fold, so this corrupts the
    fold in two ways and requires pass=False for both:

      (i)  MLP tensor scaled by 1 + 1e-7.  A relative perturbation ~100x SMALLER
           than the old absolute 1e-5 logit budget: the superseded gate would
           have waved it through, the fp64 tier must not.
      (ii) the value factors Vv rolled by one head -- a gross rewiring.

    Recorded non-corruption (a real gauge freedom, not a bug): dropping the
    0.5*(M + M^T) symmetrization in fold_mlp changes T but NOT the quadratic
    form T(z,z), so it is undetectable by construction and is not used here."""
    cfg = TFConfig(depth=depth, width=width, vocab=vocab, variant='vanilla')
    m = make_model(cfg, device)
    with torch.no_grad():
        gp = torch.Generator().manual_seed(11)
        for blk in m.h:
            for p in (blk.c_proj.weight, blk.Down.weight):
                p.copy_(torch.randn(p.shape, generator=gp) * 0.05)
            blk.Down_bias.copy_(
                torch.randn(blk.Down_bias.shape, generator=gp) * 0.01)
    idx = torch.randint(0, vocab, (2, 32), device=device)
    out = {'clean_pass': bool(check_fold_identities(m, idx,
                                                    verbose=False)['pass'])}

    def corrupt(tag, mlp_scale=1.0, roll_v=0):
        orig_mlp, orig_qk = m.fold_mlp, m.fold_layer0_qk

        def fm(*a, **kw):
            return orig_mlp(*a, **kw) * mlp_scale

        def fq(*a, **kw):
            f = orig_qk(*a, **kw)
            if roll_v:
                f['Vv'] = torch.roll(f['Vv'], roll_v, 0)
            return f
        m.fold_mlp, m.fold_layer0_qk = fm, fq
        try:
            r = check_fold_identities(m, idx, verbose=False)
        finally:
            m.fold_mlp, m.fold_layer0_qk = orig_mlp, orig_qk
        out[f'{tag}_pass'] = bool(r['pass'])
        out[f'{tag}_fp64_logit_abs'] = r['fp64_fold_forward_max_logit_diff']
        out[f'{tag}_fp32_logit_abs'] = r['fold_forward_max_logit_diff']
        out[f'{tag}_caught_by_old_absolute_1e-5_gate'] = bool(
            r['fold_forward_max_logit_diff'] >= 1e-5)

    corrupt('mlp_scale_1p1e-7', mlp_scale=1.0 + 1e-7)
    corrupt('vv_head_roll', roll_v=1)
    out['pass'] = bool(out['clean_pass']
                       and not out['mlp_scale_1p1e-7_pass']
                       and not out['vv_head_roll_pass'])
    return out


@torch.no_grad()
def planted_qk_test(vocab=64, width=32, device='cuda', tol=1e-10):
    """PLANTED KNOWN-ANSWER TEST for the table materializer.

    Build weights whose layer-0 V x V table is ANALYTICALLY known: make c_q and
    c_k rank one on head 0, W_q = alpha * u a^T, W_k = beta * w b^T.  Then
    W_q Ehn(t) = alpha (a.Ehn(t)) u and the per-head RMSNorm collapses it to
    sgn(alpha (a.Ehn(t))) * u_hat with u_hat = u*sqrt(hd)/||u||, so
        s1[t, u] = sgn_q(t) * sgn_k(u) * (u_hat^T R(delta) w_hat) / hd
    -- an outer product of two sign vectors times a scalar.  The materializer
    must reproduce it exactly.  Run in FLOAT64 on both sides: the table product
    q^T R k cancels ~2 orders of magnitude (16 terms of size ~hd -> an O(1)
    result), so fp32 floors the achievable agreement at ~5e-5 for reasons that
    have nothing to do with correctness."""
    cfg = TFConfig(depth=1, width=width, vocab=vocab, variant='vanilla')
    out = {}
    with exact_math():
        m = TinyBilin(cfg).to(device).double().eval()
        hd = cfg.head_dim
        g = torch.Generator().manual_seed(7)
        u = torch.randn(hd, generator=g, dtype=torch.float64).to(device)
        w = torch.randn(hd, generator=g, dtype=torch.float64).to(device)
        a = torch.randn(m.Ws, generator=g, dtype=torch.float64).to(device)
        b = torch.randn(m.Ws, generator=g, dtype=torch.float64).to(device)
        Wq = torch.zeros(cfg.width, m.Ws, device=device, dtype=torch.float64)
        Wk = torch.zeros(cfg.width, m.Ws, device=device, dtype=torch.float64)
        Wq[:hd] = 2.5 * torch.outer(u, a)
        Wk[:hd] = -1.3 * torch.outer(w, b)
        m.h[0].c_q.weight.copy_(Wq)
        m.h[0].c_k.weight.copy_(Wk)
        Ehn = m.token_input_table(torch.float64)
        sgn_q = torch.sign(2.5 * (Ehn @ a))
        sgn_k = torch.sign(-1.3 * (Ehn @ b))
        uh = u * math.sqrt(hd) / u.norm()
        wh = w * math.sqrt(hd) / w.norm()
        for d in (0, 3):
            # fp64 on BOTH sides: rot_matrix now honours dtype, so asking for
            # the default fp32 R here and upcasting would put an fp32-rounded
            # rotation against the fold's fp64 one and floor the test at 6e-9.
            R = rot_matrix(d, hd, device, torch.float64)
            scalar = float(uh @ R @ wh) / hd
            analytic = torch.outer(sgn_q, sgn_k) * scalar
            fl = m.fold_layer0_qk(deltas=(d,), materialize=True, heads=[0],
                                  device=device, dtype=torch.float64)
            got = fl['tables'][d]['s1'][0].to(device)
            out[f'planted_delta{d}_maxdiff'] = float((got - analytic).abs().max())
            out[f'planted_delta{d}_scale'] = abs(scalar)
    out['pass'] = all(v < tol for k, v in out.items() if k.endswith('_maxdiff'))
    return out


@torch.no_grad()
def _copy_reads(dst, src):
    dst.wte.weight.copy_(src.wte.weight)
    for bd, bs in zip(dst.h, src.h):
        for nm in READ_NAMES:
            getattr(bd, nm).weight.copy_(getattr(bs, nm).weight)


@torch.no_grad()
def variant_reduction_test(depth=1, width=32, vocab=64, device='cuda',
                           tol=1e-5):
    """Every variant with its distinguishing mechanism DISABLED must reduce to
    its parent.  Same standard as the parent program's identity controls: max
    logit difference on a fixed batch, tf32 off SYMMETRICALLY.  The predicate
    and codebook reductions are required to be BIT-EXACT (0.0) because their
    parameters are zero-init / buffer-only and consume no RNG."""
    out = {}
    with exact_math():
        idx = torch.randint(0, vocab, (2, 32), device=device)

        def build(**kw):
            cfg = TFConfig(depth=depth, width=width, vocab=vocab, **kw)
            return make_model(cfg, device).float().eval()

        # --- slots(n_slots=1, lasso 0, zero writes) == vanilla ---
        van = build(variant='vanilla')
        s1 = build(variant='slots', n_slots=1, group_coeff=0.0)
        _copy_reads(s1, van)
        for b1, b0 in zip(s1.h, van.h):
            b1.c_proj.weight.copy_(b0.c_proj.weight)
            b1.Down.weight.copy_(b0.Down.weight)
            b1.Down_bias.copy_(b0.Down_bias)
        out['slots_off_vs_vanilla'] = _maxdiff(s1(idx), van(idx))
        del van, s1
        # --- bandwidth with s = Ws/n_slots (slot rows of the full decoder) ---
        slots = build(variant='slots')
        band = build(variant='bandwidth', slot=slots.s)
        assert band.Ws == slots.Ws, (band.Ws, slots.Ws)
        _copy_reads(band, slots)
        S = slots.s
        for l, (bb, bs) in enumerate(zip(band.h, slots.h)):
            bb.c_proj.weight.copy_(bs.c_proj.weight[2 * l * S:(2 * l + 1) * S])
            bb.Down.weight.copy_(bs.Down.weight[(2 * l + 1) * S:(2 * l + 2) * S])
            bb.Down_bias.copy_(bs.Down_bias[(2 * l + 1) * S:(2 * l + 2) * S])
        out['bandwidth_slotrows_vs_slots'] = _maxdiff(band(idx), slots(idx))
        # --- predicate with terms zeroed == bandwidth (bit-exact) ---
        pred = build(variant='predicate', slot=slots.s)
        pred.load_state_dict(band.state_dict(), strict=False)
        pred.pred_prof.zero_(), pred.pred_b.zero_(), pred.pred_c.zero_()
        out['predicate_zeroed_vs_bandwidth'] = _maxdiff(pred(idx), band(idx))
        pred.pred_on = False
        out['predicate_off_vs_bandwidth'] = _maxdiff(pred(idx), band(idx))
        # --- codebook with quantization off == bandwidth (bit-exact) ---
        cb = build(variant='codebook', slot=slots.s)
        cb.load_state_dict(band.state_dict(), strict=False)
        cb.qz_on = False
        out['codebook_off_vs_bandwidth'] = _maxdiff(cb(idx), band(idx))
        # --- shrink with mode 'control' == slots ---
        sh = build(variant='shrink', shrink_mode='control')
        sh.load_state_dict(slots.state_dict(), strict=False)
        out['shrink_control_vs_slots'] = _maxdiff(sh(idx), slots(idx))
    out['pass'] = bool(
        all(v < tol for k, v in out.items() if k != 'pass')
        and out['predicate_zeroed_vs_bandwidth'] == 0.0
        and out['codebook_off_vs_bandwidth'] == 0.0)
    return out


if __name__ == '__main__':
    dev = 'cuda' if (torch.cuda.is_available()
                     and os.environ.get('TF_CPU') != '1') else 'cpu'
    print(f'== tf_model self-test on {dev} ==', flush=True)
    for depth in (1, 2):
        for width in (32, 64):
            cfg = TFConfig(depth=depth, width=width, variant='bandwidth')
            s, target, per_s = solve_slot(cfg)
            print(f'  solve_slot d{depth} w{width}: s={s} stream={s*2*depth} '
                  f'(vanilla body {target}, per_s {per_s})', flush=True)
    print('planted known-answer table test:', planted_qk_test(device=dev),
          flush=True)
    print('variant reduction battery:', variant_reduction_test(device=dev),
          flush=True)
    print('fold-gate negative control:', gate_negative_control(device=dev),
          flush=True)
    print('fold identity gate per variant (zero-init decoders PERTURBED so the '
          'gate is not vacuous):', flush=True)
    for variant in VARIANTS:
        for depth in (1, 2):
            cfg = TFConfig(depth=depth, width=32, vocab=128, variant=variant)
            m = make_model(cfg, dev)
            with torch.no_grad():                 # stand in for a trained model
                gp = torch.Generator().manual_seed(11)
                for blk in m.h:
                    for p in (blk.c_proj.weight, blk.Down.weight):
                        p.copy_(torch.randn(p.shape, generator=gp) * 0.05)
                    blk.Down_bias.copy_(
                        torch.randn(blk.Down_bias.shape, generator=gp) * 0.01)
            idx = torch.randint(0, 128, (2, 32), device=dev)
            r = check_fold_identities(m, idx, verbose=False)
            print(f'  {variant:10s} d{depth}: pass={r["pass"]}  '
                  f'attn {r["attn_layer0_table_identity_relmax"]:.2e}  '
                  f'mlp {r["mlp_tensor_identity_l0_relmax"]:.2e}  '
                  f'gauge {r["mlp_rmsnorm_gauge_l0_relmax"]:.2e}  '
                  f'logit {r["fold_forward_max_logit_diff"]:.2e}', flush=True)
            del m
