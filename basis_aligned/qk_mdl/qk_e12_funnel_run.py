"""E12 FUNNEL FAMILY, REVISED (Logan 2026-08-05 second pass; fresh
single-epoch batch-16, recipe conventions -- per-slot norm + Muon 0.02 +
in-loss lasso 3e-5 -- paired vs E9a/E0a/E0b).

CONCEPT: wide stream for detokenization, then NARROW. Embedding + block 0 run
wide; block 0's outputs are the last thing computed wide; blocks 1-11 run
narrow with NO embedding re-injection (pure write slots); block 0's
detokenized output is the token channel (a LEARNED compression -- registered
prediction: beats E4's raw-token bottleneck price of +0.105).

ARMS (Logan-spec order):
  E12L   PRIMARY: wide 384 (6 heads x 64) for embedding + block 0; narrow
         slots at 11 DIMS MATCHING E9A -- 26 slots x 11 = 286 (slightly wider
         than 264; per-module MESSAGE BANDWIDTH is what is matched, isolating
         the funnel effect from bandwidth reduction). Narrow attention 11
         heads x 26. Slots 0-1 attn0 line (22 dims via learned P_a), 2-3 mlp0
         line (P_m), 4-25 the 22 downstream modules.
  E12Lv  E12L + shared values: narrow blocks drop value projections; values
         come from block 0's WIDE value projection through learned P_sv
         (384 -> 286, reshaped 11 x 26).
  E12a   the original 264-wide -> 208-narrow (26 x 8) funnel -- measures the
         ADDITIONAL price of true narrowing.
  E12b   (only if L/Lv/a all trained stably: no divergence/starvation) the
         old 264-wide shared-values funnel. Old E12c is DROPPED -- E12L is
         the wide-segment experiment now.

ACCOUNTING (Logan): body-only parameter counts reported SEPARATELY from
embedding counts (embedding = VRAM, not capacity; the 384 tied embedding is
~19.3M vs 13.3M and is excluded from the headline comparison). Measured step
time (timed loop at the training batch) and an analytic FLOPs/token estimate
per arm with the wide block 0's extra compute called out, plus the same
estimate for E9a, so comparisons can be compute-adjusted.

FAILURE DIAGNOSTICS (on for all funnel arms):
  a. NECK INFORMATION PROBE -- the starvation-vs-optimization discriminator:
     ridge-regress the current token's normed embedding from (i) the neck
     (block-0's output slots) and (ii) the full narrow stream entry at blocks
     3/7/11; report top-1 token-recovery accuracy vs the same probe run on
     E9A's stream (its block-0 writes and entries at 3/7/11).
  b. NECK UTILIZATION: singular-value spectra of the trained P_a/P_m(/P_sv)/
     W_up -- participation-ratio effective rank + rank at 90 percent energy
     (a collapsed neck = self-inflicted bottleneck).
  c. E6-style diagnostics every 200 steps: post-clip grad norms and
     update-to-weight ratios split by segment (wide block 0 / neck
     projections / narrow blocks / readout up-projection / embedding), and
     every 1000 steps the narrow per-slot write-norm distribution (slot
     collapse watch).
  d. Early warning: held-100 at 1000 (measured in the callback) / 2000 / 4000
     with E9a's trajectory copied in for reference (E9a has no step-1000
     point -- its curve starts at 2000; noted).

CONTROLS: the exact reduction control runs at the 264-wide configuration
(D_n = 264, identity projections, E9a slot maps, 6x44 narrow heads, embedding
re-injection) == E9a architecture at init; an exact 384-wide reduction is not
possible (no 384-wide E9a twin exists), so E12L gets a structural control
(shape/visibility/finiteness) and shares the identical code path -- noted.
E12Lv with local values == E12L at init; penalty vs naive loop.
Results -> qk_e12.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, R2, DEPTH, F, nn, torch
import qk_e7_evenout_run as E7R
import qk_tokenline_probe as TP

JP = E.jpath('qk_e12.json')
GC12 = 3e-5
EXP = 4

if not hasattr(V8T, '_e11_penalty_patch'):       # same dispatch as E11
    V8T._e11_penalty_patch = True
    _prev_pen12 = V8T.group_penalty

    def _e12_dispatch(model):
        if hasattr(model, 'custom_group_penalty'):
            return model.custom_group_penalty()
        return _prev_pen12(model)
    V8T.group_penalty = _e12_dispatch


# ---------------- configurations ----------------
def cfg_L():
    if E.SMOKE:
        return dict(Dw=32, NHw=2, HDw=16, Gw=16, Dn=26, NHn=1, HDn=26,
                    Gn=26, sub_n=1, control=False)
    return dict(Dw=384, NHw=6, HDw=64, Gw=24, Dn=286, NHn=11, HDn=26,
                Gn=26, sub_n=11, control=False)


def cfg_a():
    if E.SMOKE:
        return dict(Dw=24, NHw=2, HDw=12, Gw=24, Dn=26, NHn=1, HDn=26,
                    Gn=26, sub_n=1, control=False)
    return dict(Dw=264, NHw=6, HDw=44, Gw=24, Dn=208, NHn=4, HDn=52,
                Gn=26, sub_n=8, control=False)


def cfg_control():
    if E.SMOKE:
        return dict(Dw=24, NHw=2, HDw=12, Gw=24, Dn=24, NHn=2, HDn=12,
                    Gn=24, sub_n=1, control=True)
    return dict(Dw=264, NHw=6, HDw=44, Gw=24, Dn=264, NHn=6, HDn=44,
                Gn=24, sub_n=11, control=True)


class Blk(nn.Module):
    """One bilinear block's parameters at width D (Q.Block layer order so the
    control configuration consumes the RNG identically to E1Route)."""

    def __init__(self, D):
        super().__init__()
        self.c_q = nn.Linear(D, D, bias=False)
        self.c_k = nn.Linear(D, D, bias=False)
        self.c_q2 = nn.Linear(D, D, bias=False)
        self.c_k2 = nn.Linear(D, D, bias=False)
        self.c_v = nn.Linear(D, D, bias=False)
        self.c_proj = nn.Linear(D, D, bias=False)
        self.c_proj.weight.data.zero_()
        self.Left = nn.Linear(D, EXP * D, bias=False)
        self.Right = nn.Linear(D, EXP * D, bias=False)
        self.Down = nn.Linear(EXP * D, D, bias=False)
        self.Down.weight.data.zero_()
        self.Down_bias = nn.Parameter(torch.zeros(D))


class FunnelRoute(nn.Module):
    N_SRC = 24                                   # a0n, m0n, 22 narrow writes

    def __init__(self, variant, cfg, shared_values=False, local_values=False):
        super().__init__()
        self.variant = variant
        self.cfg = cfg
        self.shared_values = shared_values and not local_values
        c = cfg
        self.Dw, self.Dn, self.Gw, self.Gn = c['Dw'], c['Dn'], c['Gw'], c['Gn']
        self.NHw, self.HDw, self.NHn, self.HDn = (c['NHw'], c['HDw'],
                                                  c['NHn'], c['HDn'])
        self.control = c['control']
        sub_n = c['sub_n']
        self.wte = nn.Embedding(Q.V, self.Dw)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.wb = Blk(self.Dw)                   # wide block 0
        self.nh = nn.ModuleList([Blk(self.Dn) for _ in range(DEPTH - 1)])
        if shared_values and not local_values:
            for blk in self.nh:
                blk.c_v = None
        gw = torch.Generator().manual_seed(888)  # nonzero write init
        std = 0.02 / math.sqrt(2 * DEPTH)
        with torch.no_grad():
            for blk in [self.wb] + list(self.nh):
                for p in (blk.c_proj.weight, blk.Down.weight):
                    p.copy_(torch.randn(p.shape, generator=gw) * std)
        self.P_a = nn.Linear(self.Dw, self.Dn, bias=False)
        self.P_m = nn.Linear(self.Dw, self.Dn, bias=False)
        self.W_up = nn.Linear(self.Dn, self.Dw, bias=False)
        self.P_sv = nn.Linear(self.Dw, self.Dn, bias=False) \
            if self.shared_values else None
        with torch.no_grad():
            if self.control:
                for lin in (self.P_a, self.P_m, self.W_up):
                    lin.weight.copy_(torch.eye(self.Dn))
            else:
                for lin in [self.P_a, self.P_m, self.W_up] \
                        + ([self.P_sv] if self.P_sv is not None else []):
                    nn.init.orthogonal_(lin.weight)
        wm_w = torch.ones(2, self.Dw)
        if self.control:
            wm_w.zero_()
            sw = self.Dw // self.Gw
            wm_w[0, 0:sw] = 1.0
            wm_w[1, sw:2 * sw] = 1.0
        self.register_buffer('wmask_w', wm_w)
        n0 = 1 if self.control else 2            # slots per block-0 line
        wm_n = torch.zeros(2 * (DEPTH - 1), self.Dn)
        src_m = torch.zeros(2, self.Dn)
        src_m[0, 0:n0 * sub_n] = 1.0
        src_m[1, n0 * sub_n:2 * n0 * sub_n] = 1.0
        for j in range(2 * (DEPTH - 1)):
            s0 = 2 * n0 * sub_n + j * sub_n
            wm_n[j, s0:s0 + sub_n] = 1.0
        self.register_buffer('wmask_n', wm_n)
        self.register_buffer('srcmask', src_m)
        cosw, sinw = Q.rope_tables_exact(Q.T, self.HDw, 'cpu')
        cosn, sinn = Q.rope_tables_exact(Q.T, self.HDn, 'cpu')
        self.register_buffer('cos_w', cosw)
        self.register_buffer('sin_w', sinw)
        self.register_buffer('cos_n', cosn)
        self.register_buffer('sin_n', sinn)
        self.register_buffer('mask',
                             torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool)))
        self.src_dims = [(0, n0 * sub_n), (n0 * sub_n, 2 * n0 * sub_n)] + \
            [(2 * n0 * sub_n + j * sub_n, 2 * n0 * sub_n + (j + 1) * sub_n)
             for j in range(2 * (DEPTH - 1))]

    def _snorm(self, x, G):
        S = x.shape[-1] // G
        sh = x.shape
        return F.rms_norm(x.view(*sh[:-1], G, S), (S,)).view(sh)

    def custom_group_penalty(self):
        tot = None
        for blk, G in [(self.wb, self.Gw)] + [(b, self.Gn) for b in self.nh]:
            for nm in E.READ_NAMES:
                lin = getattr(blk, nm, None)
                if lin is None:
                    continue
                M = lin.weight
                S = M.shape[1] // G
                g = (M.pow(2).view(M.shape[0], G, S).sum(dim=(0, 2))
                     + 1e-12).sqrt().sum()
                tot = g if tot is None else tot + g
        return tot

    def _attn(self, blk, hn, v, B, Tq, NH, HD, cos, sin):
        def qk(lin):
            z = lin(hn).view(B, Tq, NH, HD)
            return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)
        q, k = qk(blk.c_q), qk(blk.c_k)
        q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~self.mask[:Tq, :Tq], 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        return y.reshape(B, Tq, NH * HD)

    def n_visible(self, li):
        return self.N_SRC if li >= DEPTH else 2 + 2 * (li - 1)

    def forward(self, idx, collect=None, nsub=None):
        B, Tq = idx.shape
        cw = self.cos_w[None, :Tq, None, :]
        sw = self.sin_w[None, :Tq, None, :]
        cn = self.cos_n[None, :Tq, None, :]
        sn = self.sin_n[None, :Tq, None, :]
        e = F.rms_norm(self.wte(idx), (self.Dw,))
        hn = self._snorm(e, self.Gw)
        v0 = self.wb.c_v(hn)
        y = self._attn(self.wb, hn, v0.view(B, Tq, self.NHw, self.HDw),
                       B, Tq, self.NHw, self.HDw, cw, sw)
        aw0 = self.wb.c_proj(y) * self.wmask_w[0].to(y.dtype) \
            if self.control else self.wb.c_proj(y)
        x0 = e + aw0
        xn0 = self._snorm(x0, self.Gw)
        mw0 = self.wb.Down(self.wb.Left(xn0) * self.wb.Right(xn0)) \
            + self.wb.Down_bias
        if self.control:
            mw0 = mw0 * self.wmask_w[1].to(mw0.dtype)
        a0n = self.P_a(aw0) * self.srcmask[0].to(aw0.dtype)
        m0n = self.P_m(mw0) * self.srcmask[1].to(mw0.dtype)
        nsrc = [a0n, m0n]
        v_sh = None
        if self.shared_values:
            v_sh = self.P_sv(v0).view(B, Tq, self.NHn, self.HDn)
        if collect is not None:
            collect['wide_write'] = [aw0.detach(), mw0.detach()]
            if 'neck' in collect:
                collect['neck'] = (a0n + m0n).detach()

        def nentry(li):
            sub = nsub.get(li) if nsub is not None else None
            tot = e if self.control else None
            for si in range(self.n_visible(li)):
                s = sub[si] if (sub is not None and si in sub) else nsrc[si]
                tot = s if tot is None else tot + s
            return tot

        for lb, blk in enumerate(self.nh):
            li = lb + 1
            x = nentry(li)
            if collect is not None:
                collect.setdefault('entry_norm', []).append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'nentry' in collect:
                    collect['nentry'].append(x.detach())
            hnn = self._snorm(x, self.Gn)
            v = v_sh if self.shared_values else \
                blk.c_v(hnn).view(B, Tq, self.NHn, self.HDn)
            y = self._attn(blk, hnn, v, B, Tq, self.NHn, self.HDn, cn, sn)
            aw = blk.c_proj(y) * self.wmask_n[2 * lb].to(y.dtype)
            x = x + aw
            xnn = self._snorm(x, self.Gn)
            mw = blk.Down(blk.Left(xnn) * blk.Right(xnn)) + blk.Down_bias
            mw = mw * self.wmask_n[2 * lb + 1].to(mw.dtype)
            if collect is not None:
                collect.setdefault('nattn', []).append(aw.detach())
                collect.setdefault('nmlp', []).append(mw.detach())
            nsrc.append(aw)
            nsrc.append(mw)
        x = nentry(DEPTH)
        x = F.rms_norm(x, (self.Dn,))
        logits = self.W_up(x) @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e12L():
    torch.manual_seed(Q.SEED)
    return FunnelRoute('E12L', cfg_L()).to(E.DEV)


def make_e12Lv(local_values=False):
    torch.manual_seed(Q.SEED)
    return FunnelRoute('E12Lv', cfg_L(), shared_values=True,
                       local_values=local_values).to(E.DEV)


def make_e12a(control=False):
    torch.manual_seed(Q.SEED)
    return FunnelRoute('E12a', cfg_control() if control else cfg_a()).to(E.DEV)


def make_e12b(local_values=False):
    torch.manual_seed(Q.SEED)
    return FunnelRoute('E12b', cfg_a(), shared_values=True,
                       local_values=local_values).to(E.DEV)


# ---------------- controls ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    # exact reduction control at the 264-wide configuration (a 384-wide E9a
    # twin does not exist, so E12L is covered structurally -- same code path)
    ref = E7R.make_e7m1().eval().float()
    mc = make_e12a(control=True).eval().float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        d = (mc(idx) - ref(idx)).abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"control funnel(264-wide, D_n=264, identity projections, emb "
          f"re-injection)==E9a-arch at init: max |logit diff| {d:.2e}",
          flush=True)
    assert (d == 0.0) if E.SMOKE else (d < 1e-6)
    del mc, ref
    torch.cuda.empty_cache()
    # E12L structural control + penalty naive check
    mL = make_e12L().eval().float()
    out = mL(idx)
    assert torch.isfinite(out).all()
    if not E.SMOKE:
        assert mL.Dn == 286 and mL.cfg['sub_n'] == 11 and mL.Dw == 384
        assert mL.NHn * mL.HDn == 286 and mL.HDn % 2 == 0
    assert mL.n_visible(1) == 2 and mL.n_visible(DEPTH) == 24
    print(f"control E12L structural: narrow {mL.Dn} = 26 x "
          f"{mL.cfg['sub_n']}, heads {mL.NHn}x{mL.HDn}, forward finite, "
          f"vis(block1)=2 vis(readout)=24", flush=True)
    p_fast = float(mL.custom_group_penalty())
    p_naive = 0.0
    for blk, G in [(mL.wb, mL.Gw)] + [(b, mL.Gn) for b in mL.nh]:
        for nm in E.READ_NAMES:
            lin = getattr(blk, nm, None)
            if lin is None:
                continue
            M = lin.weight
            S = M.shape[1] // G
            for k in range(G):
                p_naive += float(M[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E12L penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    del mL
    torch.cuda.empty_cache()
    # E12Lv with local values == E12L at init
    ma = make_e12L().eval().float()
    mb = make_e12Lv(local_values=True).eval().float()
    d2 = (mb(idx) - ma(idx)).abs().max().item()
    print(f"control E12Lv(local values)==E12L at init: max |logit diff| "
          f"{d2:.2e}", flush=True)
    assert (d2 == 0.0) if E.SMOKE else (d2 < 1e-6)
    mbs = make_e12Lv().eval().float()
    d3 = (mbs(idx) - ma(idx)).abs().max().item()
    print(f"sanity E12Lv shared values differ: max diff {d3:.2e}", flush=True)
    assert d3 > 1e-6
    del ma, mb, mbs
    torch.cuda.empty_cache()


# ---------------- accounting ----------------
def param_split(m):
    tot = sum(p.numel() for p in m.parameters())
    emb = m.wte.weight.numel()
    return {'total': tot, 'embedding': emb, 'body': tot - emb}


def flops_per_token(m=None, e9a=False):
    """Analytic forward FLOPs/token (2*numel per linear; attention scores and
    value application at the causal mean context T/2; unembed separate)."""
    Tavg = Q.T / 2

    def attn_flops(NH, HD):
        # two score maps + value application, per token at mean context
        return 2 * Tavg * NH * HD * 2 * 2 + 2 * Tavg * NH * HD

    if e9a:
        D = Q.D
        per_block = 2 * (6 * D * D) + 2 * (2 * D * EXP * D + EXP * D * D) \
            + attn_flops(Q.NH, Q.HD)
        return {'blocks_total': round(DEPTH * per_block / 1e6, 2),
                'unembed': round(2 * D * Q.V / 1e6, 2),
                'total_Mflops_per_token': round(
                    (DEPTH * per_block + 2 * D * Q.V) / 1e6, 2)}

    def lin_f(lin):
        return 0 if lin is None else 2 * lin.weight.numel()
    names = ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'c_proj', 'Left',
             'Right', 'Down')
    wide = sum(lin_f(getattr(m.wb, nm)) for nm in names) \
        + attn_flops(m.NHw, m.HDw)
    narrow = 0
    for blk in m.nh:
        narrow += sum(lin_f(getattr(blk, nm, None)) for nm in names) \
            + attn_flops(m.NHn, m.HDn)
    neck = lin_f(m.P_a) + lin_f(m.P_m) + lin_f(m.P_sv)
    up = lin_f(m.W_up)
    unemb = 2 * m.Dw * Q.V
    return {'wide_block0': round(wide / 1e6, 2),
            'narrow_blocks_total': round(narrow / 1e6, 2),
            'neck_projections': round(neck / 1e6, 2),
            'readout_up': round(up / 1e6, 2),
            'unembed': round(unemb / 1e6, 2),
            'total_Mflops_per_token': round(
                (wide + narrow + neck + up + unemb) / 1e6, 2),
            'note': 'analytic; attention at causal mean context T/2'}


def measure_step_time(factory):
    if E.SMOKE:
        return None
    model = factory()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    order = Q.epoch_order(998)
    ts = []
    for s in range(8):
        seqs = Q.TRAIN[order[s * Q.BATCH:(s + 1) * Q.BATCH]]
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        (ce + GC12 * V8T.group_penalty(model)).backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    del model, opt
    torch.cuda.empty_cache()
    return round(sum(ts[3:]) / len(ts[3:]), 4)


# ---------------- diagnostics ----------------
def seg_params(m):
    return {'wide_block0': list(m.wb.parameters()),
            'neck': [m.P_a.weight, m.P_m.weight]
            + ([m.P_sv.weight] if m.P_sv is not None else []),
            'narrow_blocks': list(m.nh.parameters()),
            'w_up': [m.W_up.weight],
            'embedding': [m.wte.weight]}


def make_diag_cb(rec, lr_muon, lr_adamw, total_steps):
    def fac(step):
        if step < Q.WARMUP:
            return (step + 1) / Q.WARMUP
        p = (step - Q.WARMUP) / max(1, total_steps - Q.WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))

    def cb(step, model):
        with torch.no_grad():
            f = fac(step)
            d = {'step': step}
            for seg, plist in seg_params(model).items():
                gn = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum())
                                   for p in plist if p.grad is not None))
                wn = math.sqrt(sum(float(p.detach().float().pow(2).sum())
                                   for p in plist))
                lr = lr_adamw if seg == 'embedding' else lr_muon
                d[f'grad_{seg}'] = round(gn, 4)
                d[f'upd_ratio_{seg}'] = round(lr * f * gn / (wn + 1e-12), 6)
            if step % 1000 == 0:
                col = {'neck': None}
                model.eval()
                model(Q.HELD[:2, :Q.T], collect=col)
                model.train()
                sn = [round(float(col['neck'][..., d0:d1]
                                  .float().norm(dim=-1).mean()), 4)
                      for d0, d1 in model.src_dims[:2]]
                writes = []
                for lb in range(DEPTH - 1):
                    writes.append(col['nattn'][lb])
                    writes.append(col['nmlp'][lb])
                for j, w in enumerate(writes):
                    d0, d1 = model.src_dims[2 + j]
                    sn.append(round(float(
                        w[..., d0:d1].float().norm(dim=-1).mean()), 4))
                d['narrow_slot_write_norms'] = sn
                d['collapsed_slots_below_1e-3'] = sum(1 for x in sn
                                                      if x < 1e-3)
                del col
            if step == 1000 and not E.SMOKE:
                model.eval()
                hce, _ = Q.eval_held(model, n_seq=100)
                model.train()
                d['held100_fresh'] = round(hce, 4)
                print(f"  E12 diag step 1000 held100 {hce:.4f}", flush=True)
            rec.append(d)
    cb.every = 200
    return cb


@torch.no_grad()
def neck_spectra(m):
    out = {}
    for nm in ('P_a', 'P_m', 'P_sv', 'W_up'):
        lin = getattr(m, nm, None)
        if lin is None:
            continue
        sv = torch.linalg.svdvals(lin.weight.detach().float()).cpu()
        eff = float(sv.sum() ** 2 / sv.pow(2).sum())
        energy = sv.pow(2).cumsum(0) / sv.pow(2).sum()
        out[nm] = {'top20_sv': [round(float(x), 4) for x in sv[:20]],
                   'effective_rank_participation': round(eff, 2),
                   'rank_at_90pct_energy': int((energy < 0.9).sum()) + 1,
                   'n_sv': int(len(sv))}
    return out


N_FIT, N_EV = 48, 16


@torch.no_grad()
def _ridge_top1(H_fit, ids_fit, H_ev, ids_ev, wte_normed):
    H = H_fit.double()
    lam = 1e-3 * float(H.pow(2).mean()) * H.shape[1]
    A = H.t() @ H + lam * torch.eye(H.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, H.t() @ wte_normed[ids_fit].double())
    pred = (H_ev.double() @ W).float()
    hits, n = 0, 0
    for i in range(0, len(pred), 1024):
        lg = pred[i:i + 1024] @ wte_normed.t()
        hits += int((lg.argmax(1) == ids_ev[i:i + 1024]).sum())
        n += len(lg)
    return hits / max(1, n)


@torch.no_grad()
def neck_info_probe(m, funnel=True):
    """Top-1 token recovery by ridge regression from (neck, stream@3/7/11)."""
    m.eval().float()
    Dv = m.wte.weight.shape[1]
    wten = F.rms_norm(m.wte.weight.detach().float(), (Dv,)).cpu()
    sites = {'neck_block0_outputs': [], 'stream_block3': [],
             'stream_block7': [], 'stream_block11': []}
    ids_all = []
    for i in range(0, N_FIT + N_EV, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        if funnel:
            col = {'neck': None, 'nentry': []}
            m(b, collect=col)
            sites['neck_block0_outputs'].append(
                col['neck'].float().reshape(-1, m.Dn).cpu())
            for nm, li in (('stream_block3', 3), ('stream_block7', 7),
                           ('stream_block11', 11)):
                sites[nm].append(
                    col['nentry'][li - 1].float().reshape(-1, m.Dn).cpu())
        else:
            col = {'entry': [], 'entry_norm': [], 'attn_write': [],
                   'mlp_write': []}
            m(b, collect=col)
            neck = (col['attn_write'][0] + col['mlp_write'][0]).float()
            sites['neck_block0_outputs'].append(
                neck.reshape(-1, neck.shape[-1]).cpu())
            for nm, li in (('stream_block3', 3), ('stream_block7', 7),
                           ('stream_block11', 11)):
                sites[nm].append(col['entry'][li].float()
                                 .reshape(-1, col['entry'][li].shape[-1])
                                 .cpu())
        ids_all.append(b.reshape(-1).cpu())
        del col
        torch.cuda.empty_cache()
    ids = torch.cat(ids_all)
    nfit = N_FIT * Q.T
    res = {}
    for nm, chunks in sites.items():
        H = torch.cat(chunks)
        acc = _ridge_top1(H[:nfit], ids[:nfit], H[nfit:], ids[nfit:], wten)
        res[nm] = round(acc, 4)
        print(f"  neck probe {nm}: top-1 {acc:.4f}", flush=True)
    res['note'] = ('ridge from representation to the normed embedding, '
                   'argmax over vocab; fit 48 held seqs, eval 16')
    return res


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


# ---------------- funnel probes (wiring + token-determined) ----------------
ABL = 8 if E.SMOKE else 96
SRC_NAMES = ['attn0n', 'mlp0n'] + [f'{k}{l}' for l in range(1, DEPTH)
                                   for k in ('attn', 'mlp')]


@torch.no_grad()
def _base_and_nmeans(m):
    tot, n, pt = None, 0, []
    for i in range(0, ABL, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {'neck': None}
        logits = m(b, collect=col)
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        pt.append(ce.cpu())
        a0n = m.P_a(col['wide_write'][0]) * m.srcmask[0]
        m0n = m.P_m(col['wide_write'][1]) * m.srcmask[1]
        ordered = [a0n, m0n]
        for lb in range(DEPTH - 1):
            ordered.append(col['nattn'][lb])
            ordered.append(col['nmlp'][lb])
        if tot is None:
            tot = [torch.zeros(m.Dn, dtype=torch.float64, device=b.device)
                   for _ in range(m.N_SRC)]
        for si in range(m.N_SRC):
            tot[si] += ordered[si].double().sum((0, 1))
        n += b.shape[0] * Q.T
    return float(torch.cat(pt).mean()), [(t / n).float() for t in tot]


@torch.no_grad()
def _ce_with(m, nsub):
    tot, n = 0.0, 0
    for i in range(0, ABL, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        B = b.shape[0]
        se = {li: {si: mv[None, None, :].expand(B, Q.T, m.Dn)
                   for si, mv in d.items()} for li, d in nsub.items()}
        logits = m(b, nsub=se)
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             b[:, 1:Q.T + 1].reshape(-1), reduction='none')
        tot += ce.sum().item()
        n += ce.numel()
    return tot / n


def funnel_light_probe(m):
    m.eval().float()
    base, means = _base_and_nmeans(m)
    dce = {}
    t0 = time.time()
    for li in list(range(1, DEPTH)) + [DEPTH]:
        dce[li] = {}
        for si in range(m.n_visible(li)):
            dce[li][si] = _ce_with(m, {li: {si: means[si]}}) - base
        top = sorted(dce[li], key=lambda s: -dce[li][s])[:3]
        print(f"  funnel consumption row {li}: " + ", ".join(
            f"{SRC_NAMES[s]} {dce[li][s]:+.4f}" for s in top)
            + f" ({time.time() - t0:.0f}s)", flush=True)
    sup = {}
    for li in list(range(1, DEPTH)) + [DEPTH]:
        row = {}
        for si in range(m.n_visible(li)):
            d0, d1 = m.src_dims[si]
            if li == DEPTH:
                row[si] = float(m.W_up.weight[:, d0:d1].float()
                                .pow(2).sum()) ** 0.5
            else:
                blk = m.nh[li - 1]
                row[si] = float(sum(
                    getattr(blk, nm).weight[:, d0:d1].float().pow(2).sum()
                    for nm in E.READ_NAMES
                    if getattr(blk, nm, None) is not None)) ** 0.5
        sup[li] = row
    wp = [(li, si) for li in sup for si in sup[li]]
    sv = [sup[li][si] for li, si in wp]
    cv = [dce[li][si] for li, si in wp]
    eff = [k for k in range(len(wp)) if cv[k] > 0.005]
    top_s = set(np.argsort(sv)[::-1][:10].tolist())
    top_c = set(np.argsort(cv)[::-1][:10].tolist())
    r = {'base_ce_fp32_abl_fresh': round(base, 5),
         'wiring_n_pairs': len(wp),
         'wiring_spearman_all': round(R2.spearman(sv, cv), 4),
         'wiring_n_effectual': len(eff),
         'wiring_spearman_effectual': round(R2.spearman(
             [sv[k] for k in eff], [cv[k] for k in eff]), 4),
         'wiring_top10_precision': len(top_s & top_c) / 10.0,
         'consumption_matrix': {str(li): {SRC_NAMES[si]: round(v, 6)
                                          for si, v in dce[li].items()}
                                for li in dce},
         'note': 'custom funnel probe; ablation means on FRESH held; readout '
                 'support = W_up column groups'}
    print(f"funnel wiring: Spearman all {r['wiring_spearman_all']} "
          f"effectual({len(eff)}) {r['wiring_spearman_effectual']} "
          f"top10 {r['wiring_top10_precision']}", flush=True)
    return r


@torch.no_grad()
def funnel_tok_probe(m):
    m.eval().float()
    N = 16 if E.SMOKE else 200
    mws, wides = [], []
    for i in range(0, N, 8):
        b = Q.HELD[i:i + 8, :Q.T]
        col = {}
        m(b, collect=col)
        wides.append(col['wide_write'][1].float().cpu())
        mws.append(torch.stack([w.float().cpu() for w in col['nmlp']], 1))
    mw = torch.cat(mws)
    wd = torch.cat(wides)
    gids = Q.HELD[:N, :Q.T].reshape(-1).cpu().numpy().astype(np.int64)
    elig = np.ones(len(gids), bool)
    out = []
    Xw = wd.reshape(-1, m.Dw).double()
    r2, _, cov = TP.group_r2(Xw, gids, elig, 5)
    out.append({'module': 'mlp0_wide', 'r2': round(r2, 4),
                'shuffle_ctl': round(TP.shuffle_control(Xw, gids, cov), 4)})
    for lb in range(DEPTH - 1):
        X = mw[:, lb, :].reshape(-1, m.Dn).double()
        if float((X - X.mean(0)).pow(2).sum()) < 1e-10:
            out.append({'module': f'mlp{lb + 1}', 'r2': None,
                        'constant_write': True})
            continue
        r2, _, cov = TP.group_r2(X, gids, elig, 5)
        out.append({'module': f'mlp{lb + 1}', 'r2': round(r2, 4),
                    'shuffle_ctl': round(TP.shuffle_control(X, gids, cov), 4)})
        print(f"  funnel tokdet mlp{lb + 1}: r2 {r2:.4f}", flush=True)
    return {'token_determined_mlp': out,
            'note': 'group-mean R^2 + shuffle control only; linear-in-'
                    'embedding causal recovery omitted (two-width arch)'}


def held_at(key, step):
    rec = E.loadj(JP).get(key, {})
    for s, ce in rec.get('held100_fresh_curve', []):
        if s == step:
            return ce
    return None


def run_arm(stem, key, factory, design, extra_pairs=()):
    diag = []
    mlr = E7R.muon_lr()
    cb = make_diag_cb(diag, mlr, E.get_lr(), V8T.STEPS)
    E.train_arm(stem, JP, key, factory, GC12, lr=mlr,
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), step_cb=cb, **kw),
                extra={'optimizer': 'muon', 'design': design})
    if diag:
        E.merge(JP, f'diag_{key}', diag)
    h1000 = next((d.get('held100_fresh') for d in diag
                  if d.get('step') == 1000), None)
    ew = {'held100_at_1000': h1000, 'held100_at_2000': held_at(key, 2000),
          'held100_at_4000': held_at(key, 4000)}
    if not E.SMOKE:
        e9a = E.loadj(E.jpath('qk_e9.json')).get('E9a', {})
        ew['e9a_reference_curve'] = e9a.get('held100_fresh_curve')
        ew['note'] = 'E9a has no step-1000 point (its curve starts at 2000)'
    E.merge(JP, f'early_warning_{key}', ew)
    rec = E.loadj(JP).get(key, {})
    h2 = held_at(key, 2000)
    if h2 is not None:
        rec['token_starvation_flag'] = bool(h2 >= 6.5)
        E.merge(JP, key, rec)
    E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
    E.paired_fresh(stem, JP, key)
    pair_extra(stem, key, (('qk_e9_a', 'e9a'),) + tuple(extra_pairs))
    if not E.SMOKE and os.path.exists(E.ckpath(stem)):
        m = None

        def need(k):
            return k not in E.loadj(JP)
        if need(f'light_probe_{key}'):
            m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'light_probe_{key}', funnel_light_probe(m))
        if need(f'tok_probe_{key}'):
            if m is None:
                m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'tok_probe_{key}', funnel_tok_probe(m))
        if need(f'neck_spectra_{key}'):
            if m is None:
                m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'neck_spectra_{key}', neck_spectra(m))
        if need(f'neck_info_probe_{key}'):
            if m is None:
                m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'neck_info_probe_{key}', neck_info_probe(m))
        if m is not None:
            del m
            torch.cuda.empty_cache()
        if need(f'step_time_{key}'):
            E.merge(JP, f'step_time_{key}',
                    {'sec_per_step_batch16': measure_step_time(factory)})


if __name__ == '__main__':
    E.setup()
    controls()

    if 'accounting' not in E.loadj(JP):
        ref = E7R.make_e7m1()
        acc = {'e9a': dict(param_split(ref), flops=flops_per_token(e9a=True)),
               'headline_note': 'body-only params are the capacity '
                                'comparison; embedding params are VRAM '
                                '(Logan), reported separately'}
        del ref
        torch.cuda.empty_cache()
        for nm, mk in (('E12L', make_e12L), ('E12Lv', make_e12Lv),
                       ('E12a', make_e12a), ('E12b', make_e12b)):
            m = mk()
            acc[nm] = dict(param_split(m), flops=flops_per_token(m))
            del m
            torch.cuda.empty_cache()
        acc['predictions'] = ['beats E4 raw-token bottleneck price (+0.105)',
                              'downstream token-determined R^2 drops vs E9a']
        E.merge(JP, 'accounting', acc)

    if not E.SMOKE and 'e9a_neck_info_reference' not in E.loadj(JP) \
            and os.path.exists(E.ckpath('qk_e9_a')):
        m9, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
        E.merge(JP, 'e9a_neck_info_reference',
                neck_info_probe(m9, funnel=False))
        del m9
        torch.cuda.empty_cache()

    run_arm('qk_e12_L', 'E12L', make_e12L,
            'funnel PRIMARY: wide 384 (6x64) detokenization, narrow 286 = '
            '26 slots x 11 (E9a-matched message bandwidth), heads 11x26')
    run_arm('qk_e12_Lv', 'E12Lv', make_e12Lv,
            'E12L + shared values from block-0 wide c_v via P_sv (384->286, '
            '11x26)', extra_pairs=(('qk_e12_L', 'e12L'),))
    run_arm('qk_e12_a', 'E12a', make_e12a,
            'funnel narrowing arm: wide 264, narrow 208 = 26 x 8 (heads '
            '4x52) -- the additional price of true narrowing',
            extra_pairs=(('qk_e12_L', 'e12L'),))

    ok = True
    for key in ('E12L', 'E12Lv', 'E12a'):
        rec = E.loadj(JP).get(key, {})
        if rec.get('diverged') or rec.get('token_starvation_flag'):
            ok = False
    if ok:
        run_arm('qk_e12_b', 'E12b', make_e12b,
                'old shared-values narrowing arm (264 wide -> 208 narrow), '
                'kept because the GPU is free after everything',
                extra_pairs=(('qk_e12_a', 'e12a'),))
    else:
        E.merge(JP, 'E12b_skipped',
                'gate: divergence or token starvation among E12L/E12Lv/E12a')
    E.merge(JP, 'E12c_dropped',
            'old wide-384 narrowing arm superseded by the primary E12L')
    print('e12 funnel run done', flush=True)
