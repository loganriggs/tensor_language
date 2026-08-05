"""E15 EFFECTIVE PARAMETERS + REINVESTMENT (Logan 2026-08-06: masked write
projections mean slotted models have far fewer EFFECTIVE parameters than
nominal -- a module's c_proj is effectively 264->11 and its Down 1056->11.
Fresh single-epoch batch-16, recipe conventions).

E15a -- EFFECTIVE-PARAMETER RECOUNT (probe, instant): for every width-264
  fresh arm whose architecture is importable, count masked-to-zero write-
  projection rows as absent. Reports nominal vs effective BODY params, the
  implied param-deficit CE via the width-sweep exchange rate
  (deficit_ce = 0.74 * ln(nominal_vanilla_body / effective_body) / ln 19),
  and where a fresh paired delta vs vanilla exists, the ACCOUNTING-ADJUSTED
  partition cost (raw minus deficit). Headline (standard 24x11 slots): waste
  4,007,520 body params, effective 11.05M vs vanilla 15.06M, deficit ~0.078
  nats -- of E9a's +0.203 vs vanilla, ~0.125 survives adjustment.

E15b -- TRUE-SMALL DECODERS + REINVESTMENT (train): write projections at
  their real size (c_proj 264->11, Down hidden->11, bias 11, all scattered
  into the module's slot), and the saved ~4M params spent growing every
  module's bilinear hidden width equally to match VANILLA's effective body
  params (the exact hidden width is solved from live counts and documented;
  attention head dims unchanged -- reinvestment goes only into MLP hidden).
  Full readable recipe, paired vs E9a AND vs E0a: THE effective-param-matched
  partition-cost measurement. REGISTERED PREDICTION: the partition cost
  shrinks materially (the accounting implies ~0.06-0.08 of the width-264 gap
  is param deficit). Identity control: the small-decoder model at the
  ORIGINAL hidden width, with weights copied from an E9a-init model (small
  decoder rows = the masked full decoder's slot rows), reproduces E9a's
  forward at init (asserted < 1e-4: the algebraic identity is exact but the
  differently-shaped GEMMs may differ in reduction order; tf32 disabled
  around the check; measured value printed).

E15c -- BANDWIDTH REINVESTMENT (runs last; GPU is free after the chain):
  same true-small decoders but the savings buy SLOT SIZE at fixed hidden
  width (1056) and fixed compute width (264, heads 6x44): the residual
  stream widens to 24 x s where s is solved so body params match vanilla
  (documented; the tied embedding widens too -- VRAM, excluded from body
  accounting). Tests reinvestment LOCATION: hidden capacity vs communication
  bandwidth (ties into the E14 saturation question). Same state-copy
  identity control at s=11.

Measured step time recorded per trained arm (true-small decoders should be
FASTER). Standard conventions; results -> qk_e15.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e15.json')
GC15 = 3e-5
EXCHANGE = (0.74, 19.0)                          # 0.74 nats per 19x params

if not hasattr(V8T, '_e11_penalty_patch'):       # custom-penalty dispatch
    V8T._e11_penalty_patch = True
    _prev_pen15 = V8T.group_penalty

    def _e15_dispatch(model):
        if hasattr(model, 'custom_group_penalty'):
            return model.custom_group_penalty()
        return _prev_pen15(model)
    V8T.group_penalty = _e15_dispatch


# ---------------- E15a: effective recount ----------------
@torch.no_grad()
def effective_body(m):
    """Nominal body params and effective body params (masked write-projection
    rows counted as absent). Generic over the V8Route family via wmask."""
    tot = sum(p.numel() for p in m.parameters())
    emb = m.wte.weight.numel()
    body = tot - emb
    waste = 0
    if getattr(m, 'proj', False) and hasattr(m, 'wmask'):
        for k in range(2 * DEPTH):
            blk = m.h[k // 2]
            W = blk.c_proj.weight if k % 2 == 0 else blk.Down.weight
            if W.shape[0] != m.wmask.shape[1]:
                continue                         # already-small decoder
            dead_rows = int((m.wmask[k] == 0).sum())
            waste += dead_rows * W.shape[1]
            if k % 2 == 1:                       # Down_bias dead entries
                waste += dead_rows
    return body, body - waste


def deficit_ce(nom_vanilla, eff):
    a, b = EXCHANGE
    return a * math.log(nom_vanilla / eff) / math.log(b)


def e15a_recount():
    if 'E15a_recount' in E.loadj(JP):
        return
    import qk_v10v11_common as W
    import qk_v13_common as X
    import qk_e2_cprank_run as E2R
    import qk_e4_tokenslot_run as E4R
    import qk_e9_compose_run as E9R
    import qk_e10_embsplit_run as E10R
    import qk_e11_lit_run as E11R
    arms = [
        ('E0a_vanilla', E.make_e0a, None, None),
        ('E0b', E.make_e0b, 'qk_e0.json', 'E0b_minus_E0a_fresh'),
        ('E5slots_arch=E0b', E.make_e0b, 'qk_e5.json',
         'E5slots_minus_e0a_fresh'),
        ('E1', E1R.make_e1, 'qk_e1.json', 'E1_minus_e0a_fresh'),
        ('E2r8', lambda: E2R.make_e2(8, 'E2r8'), 'qk_e2.json',
         'E2r8_minus_e0a_fresh'),
        ('E2r32', lambda: E2R.make_e2(32, 'E2r32'), 'qk_e2.json',
         'E2r32_minus_e0a_fresh'),
        ('E4', E4R.make_e4, 'qk_e4.json', 'E4_minus_e0a_fresh'),
        ('E9a_arch=E7m1', E7R.make_e7m1, 'qk_e9.json',
         'E9a_minus_e0a_fresh'),
        ('E9b', E9R.make_e9b, 'qk_e9.json', 'E9b_minus_e0a_fresh'),
        ('E9c', E9R.make_e9c, 'qk_e9.json', 'E9c_minus_e0a_fresh'),
        ('ER_V10', W.make_v10, 'qk_er.json', 'V10_minus_e0a_fresh'),
        ('ER_V11', lambda: W.make_v11('V11', dec_lasso=True,
                                      cls=W.V11Route),
         'qk_er.json', 'V11_minus_e0a_fresh'),
        ('ER_V11nl', lambda: W.make_v11('V11nl', dec_lasso=False,
                                        cls=W.V11Route),
         'qk_er.json', 'V11nl_minus_e0a_fresh'),
        ('ER_V13r1', lambda: X.make_v13('V13r1', 1), 'qk_er.json',
         'V13r1_minus_e0a_fresh'),
        ('E10a', E10R.make_e10a, 'qk_e10.json', 'E10a_minus_e0a_fresh'),
        ('E10b', E10R.make_e10b, 'qk_e10.json', 'E10b_minus_e0a_fresh'),
        ('E11a', E11R.make_e11a, 'qk_e11.json', 'E11a_minus_e0a_fresh'),
        ('E11b', E11R.make_e11b, 'qk_e11.json', 'E11b_minus_e0a_fresh'),
    ]
    mv = E.make_e0a()
    nom_v, _ = effective_body(mv)
    del mv
    torch.cuda.empty_cache()
    rows = []
    for name, factory, jf, jkey in arms:
        try:
            m = factory()
        except Exception as ex:
            rows.append({'arm': name, 'error': str(ex)[:120]})
            continue
        nom, eff = effective_body(m)
        del m
        torch.cuda.empty_cache()
        dce = deficit_ce(nom_v, eff)
        row = {'arm': name, 'nominal_body': nom, 'effective_body': eff,
               'masked_away': nom - eff,
               'param_deficit_ce': round(dce, 4)}
        if jf is not None and not E.SMOKE:
            d = E.loadj(E.jpath(jf)).get(jkey, {})
            raw = d.get('minus_e0a')
            if raw is not None:
                row['raw_minus_e0a'] = round(raw, 4)
                row['accounting_adjusted_partition_cost'] = round(
                    raw - dce, 4)
        rows.append(row)
        print(f"E15a {name}: body {nom} -> eff {eff} "
              f"(deficit_ce {dce:.4f})", flush=True)
    E.merge(JP, 'E15a_recount', {
        'vanilla_nominal_body': nom_v,
        'exchange_rate': '0.74 nats per 19x params (width sweep)',
        'arms': rows,
        'headline': 'standard 24x11 slots mask away 4,007,520 body params; '
                    'deficit ~0.078 nats of the measured partition costs'})


# ---------------- E15b: true-small decoders + hidden reinvestment ----------------
def solve_hidden(slot):
    """Hidden width H s.t. small-decoder body params match vanilla body."""
    D = Q.D
    mv = E.make_e0a()
    target, _ = effective_body(mv)
    del mv
    torch.cuda.empty_cache()
    per_blk_fixed = 5 * D * D + slot * D + slot  # attn reads + c_proj + bias
    H = int(round((target / DEPTH - per_blk_fixed) / (2 * D + slot)))
    return H, target


class E15bRoute(E1R.E1Route):
    """Readable recipe with TRUE-SMALL write decoders (c_proj D->slot, Down
    hidden->slot, bias slot; writes scattered into the module's slot dims)
    and hidden width `hidden` for every bilinear MLP."""

    def __init__(self, variant, depth, hidden, slot=None):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        S = slot or Dm // (2 * depth)
        self.slot = S
        self.hidden = hidden
        gw = torch.Generator().manual_seed(888)
        std = 0.02 / math.sqrt(2 * depth)
        with torch.no_grad():
            for blk in self.h:
                blk.c_proj = nn.Linear(Dm, S, bias=False)
                blk.Left = nn.Linear(Dm, hidden, bias=False)
                blk.Right = nn.Linear(Dm, hidden, bias=False)
                blk.Down = nn.Linear(hidden, S, bias=False)
                blk.Down_bias = nn.Parameter(torch.zeros(S))
                for p in (blk.c_proj.weight, blk.Down.weight):
                    p.copy_(torch.randn(p.shape, generator=gw) * std)

    def _scatter(self, small, k, B, Tq, Dm):
        out = torch.zeros(B, Tq, Dm, device=small.device, dtype=small.dtype)
        out[..., self.slot * k:self.slot * (k + 1)] = small
        return out

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = self._scatter(blk.c_proj(y), 2 * l, B, Tq, Dm)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)
                mw_s = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                mw = self._scatter(mw_s, 2 * l + 1, B, Tq, Dm)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e15b(hidden=None):
    C.register('E15b')
    S = Q.D // E.NGROUP
    if hidden is None:
        hidden, _ = solve_hidden(S)
    torch.manual_seed(Q.SEED)
    m = E15bRoute('E15b', DEPTH, hidden, slot=S).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


@torch.no_grad()
def copy_from_ref(m, ref):
    """Small decoders <- slot rows of the reference's masked full decoders."""
    S = m.slot
    m.wte.weight.copy_(ref.wte.weight)
    for l, (blk, rblk) in enumerate(zip(m.h, ref.h)):
        for nm in ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'Left', 'Right'):
            getattr(blk, nm).weight.copy_(getattr(rblk, nm).weight)
        a1 = 2 * l * S
        a2 = (2 * l + 1) * S
        blk.c_proj.weight.copy_(rblk.c_proj.weight[a1:a1 + S, :])
        blk.Down.weight.copy_(rblk.Down.weight[a2:a2 + S, :])
        blk.Down_bias.copy_(rblk.Down_bias[a2:a2 + S])


# ---------------- E15c: bandwidth reinvestment ----------------
def solve_slot_c(hidden):
    """Slot size s at fixed hidden/compute width s.t. body matches vanilla."""
    D = Q.D
    mv = E.make_e0a()
    target, _ = effective_body(mv)
    del mv
    torch.cuda.empty_cache()
    per_s = 24 * (5 * D + 2 * hidden) + D + hidden + 1
    return max(1, int(round(target / (DEPTH * per_s)))), target


class E15cRoute(nn.Module):
    """Stream width Ws = 24*s; compute width Dc (attention 6x44, hidden
    fixed); true-small decoders scatter each write into its s-dim slot."""

    def __init__(self, variant, depth, s, Dc, NH, HD, hidden):
        super().__init__()
        self.variant = variant
        self.depth = depth
        self.s, self.Dc, self.NH, self.HD = s, Dc, NH, HD
        self.slot = s                            # copy_from_ref compatibility
        self.hidden = hidden
        self.Ws = 2 * depth * s
        self.wte = nn.Embedding(Q.V, self.Ws)
        nn.init.normal_(self.wte.weight, std=0.02)

        class B(nn.Module):
            def __init__(bs):
                super().__init__()
                bs.c_q = nn.Linear(self.Ws, Dc, bias=False)
                bs.c_k = nn.Linear(self.Ws, Dc, bias=False)
                bs.c_q2 = nn.Linear(self.Ws, Dc, bias=False)
                bs.c_k2 = nn.Linear(self.Ws, Dc, bias=False)
                bs.c_v = nn.Linear(self.Ws, Dc, bias=False)
                bs.c_proj = nn.Linear(Dc, s, bias=False)
                bs.Left = nn.Linear(self.Ws, hidden, bias=False)
                bs.Right = nn.Linear(self.Ws, hidden, bias=False)
                bs.Down = nn.Linear(hidden, s, bias=False)
                bs.Down_bias = nn.Parameter(torch.zeros(s))
        self.h = nn.ModuleList([B() for _ in range(depth)])
        gw = torch.Generator().manual_seed(888)
        std = 0.02 / math.sqrt(2 * depth)
        with torch.no_grad():
            for blk in self.h:
                for p in (blk.c_proj.weight, blk.Down.weight):
                    p.copy_(torch.randn(p.shape, generator=gw) * std)
        self.vis = [list(range(1 + 2 * min(li, depth)))
                    for li in range(depth + 1)]
        ind = torch.zeros(self.Ws, 2 * depth)
        for k in range(2 * depth):
            ind[s * k:s * (k + 1), k] = 1.0
        self.register_buffer('seg_ind', ind)
        cos, sin = Q.rope_tables_exact(Q.T, HD, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask',
                             torch.tril(torch.ones(Q.T, Q.T, dtype=torch.bool)))

    def slot_norm(self, x):
        G, S = 2 * self.depth, self.s
        sh = x.shape
        return F.rms_norm(x.view(*sh[:-1], G, S), (S,)).view(sh)

    def custom_group_penalty(self):
        tot = None
        for blk in self.h:
            for nm in E.READ_NAMES:
                M = getattr(blk, nm).weight
                colsq = M.pow(2).sum(0)
                g = (colsq @ self.seg_ind + 1e-12).sqrt().sum()
                tot = g if tot is None else tot + g
        return tot

    def forward(self, idx, collect=None, sub_entry=None):
        B, Tq = idx.shape
        Ws, s = self.Ws, self.s
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        e = F.rms_norm(self.wte(idx), (Ws,))
        streams = [e]

        def entry(li):
            sub = sub_entry.get(li) if sub_entry is not None else None
            tot = None
            for i in self.vis[li]:
                v = sub[i] if (sub is not None and i in sub) else streams[i]
                tot = v if tot is None else tot + v
            return tot

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect.setdefault('entry_norm', []).append(
                    x.detach().float().norm(dim=-1).mean().item())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, self.NH, self.HD)
                return Q.apply_rot(F.rms_norm(z, (self.HD,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, self.NH, self.HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / self.HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            y = y.reshape(B, Tq, self.Dc)
            aw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            aw[..., s * 2 * l:s * (2 * l + 1)] = blk.c_proj(y)
            x = x + aw
            xn = self.slot_norm(x)
            mw_s = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
            mw = torch.zeros(B, Tq, Ws, device=y.device, dtype=y.dtype)
            mw[..., s * (2 * l + 1):s * (2 * l + 2)] = mw_s
            if collect is not None:
                collect.setdefault('attn_write', []).append(aw.detach())
                collect.setdefault('mlp_write', []).append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = F.rms_norm(entry(self.depth), (Ws,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e15c(s=None):
    hidden = 4 * Q.D
    if s is None:
        s, _ = solve_slot_c(hidden)
    torch.manual_seed(Q.SEED)
    return E15cRoute('E15c', DEPTH, s, Q.D, Q.NH, Q.HD, hidden).to(E.DEV)


# ---------------- controls ----------------
def controls():
    idx = Q.HELD[:2, :Q.T]
    S = Q.D // E.NGROUP
    ref = E7R.make_e7m1().eval().float()
    with torch.no_grad():
        # E15b at ORIGINAL hidden width, weights copied from the E9a init
        mb = make_e15b(hidden=4 * Q.D).eval().float()
        copy_from_ref(mb, ref)
        # tf32 must be off for the REFERENCE forward too: an asymmetric
        # comparison (ref in tf32, candidate in fp32) measures tf32 rounding,
        # not the architecture (6e-4 observed; symmetric fp32 gives 2e-6).
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            out_ref = ref(idx)
            d = (mb(idx) - out_ref).abs().max().item()
            # E15c at s = slotdim (Ws = D), same copy
            mc = make_e15c(s=S).eval().float()
            copy_from_ref(mc, ref)
            d2 = (mc(idx) - out_ref).abs().max().item()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"control E15b(hidden={4 * Q.D}, copied weights)==E9a at init: "
          f"max |logit diff| {d:.2e} (algebraic identity; different GEMM "
          f"shapes may differ in reduction order)", flush=True)
    assert d < 1e-4
    print(f"control E15c(s={S}, copied weights)==E9a at init: "
          f"max |logit diff| {d2:.2e}", flush=True)
    assert d2 < 1e-4
    # penalty naive check on E15c (variable machinery)
    with torch.no_grad():
        p_fast = float(mc.custom_group_penalty())
        p_naive = 0.0
        for blk in mc.h:
            for nm in E.READ_NAMES:
                M = getattr(blk, nm).weight
                for k in range(2 * DEPTH):
                    p_naive += float(
                        M[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
        rel = abs(p_fast - p_naive) / p_naive
    print(f"control E15c penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    del ref, mb, mc, out_ref
    torch.cuda.empty_cache()


def measure_step_time(factory):
    if E.SMOKE:
        return None
    model = factory()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    order = Q.epoch_order(997)
    ts = []
    for st in range(8):
        seqs = Q.TRAIN[order[st * Q.BATCH:(st + 1) * Q.BATCH]]
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        (ce + GC15 * V8T.group_penalty(model)).backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    del model, opt
    torch.cuda.empty_cache()
    return round(sum(ts[3:]) / len(ts[3:]), 4)


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


if __name__ == '__main__':
    E.setup()
    controls()
    e15a_recount()

    # ---- E15b ----
    S = Q.D // E.NGROUP
    H, target = solve_hidden(S)
    m = make_e15b(hidden=H)
    nom, eff = effective_body(m)
    E.merge(JP, 'E15b_config', {
        'hidden': H, 'slot': S, 'vanilla_body_target': target,
        'body': nom, 'effective_body': eff,
        'note': 'true-small decoders: nominal == effective by construction; '
                'reinvestment only in MLP hidden width (attention unchanged)',
        'prediction': 'partition cost vs vanilla shrinks materially '
                      '(~0.06-0.08 of the width-264 gap is param deficit)'})
    del m
    torch.cuda.empty_cache()
    E.train_arm('qk_e15_b', JP, 'E15b', lambda: make_e15b(hidden=H), GC15,
                lr=E7R.muon_lr(),
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'hidden': H,
                       'design': 'true-small decoders + hidden reinvestment '
                                 'to vanilla effective body params'})
    E.oldheld_record('qk_e15_b', lambda: make_e15b(hidden=H), JP,
                     'E15b_oldheld')
    E.paired_fresh('qk_e15_b', JP, 'E15b')
    pair_extra('qk_e15_b', 'E15b', (('qk_e9_a', 'e9a'),))
    E.probe_arm('qk_e15_b', lambda: make_e15b(hidden=H), JP,
                'light_probe_E15b', tok_key='tok_probe_E15b')
    if not E.SMOKE and 'step_time_E15b' not in E.loadj(JP):
        E.merge(JP, 'step_time_E15b', {
            'sec_per_step_batch16': measure_step_time(
                lambda: make_e15b(hidden=H)),
            'e9a_reference': 'compare qk_e_batch.json 0.1317 (full-decoder '
                             'E0b preflight)'})

    # ---- E15c ----
    s_c, tgt = solve_slot_c(4 * Q.D)
    E.merge(JP, 'E15c_config', {
        'slot': s_c, 'stream_width': 2 * DEPTH * s_c,
        'compute_width': Q.D, 'hidden': 4 * Q.D,
        'vanilla_body_target': tgt,
        'note': 'bandwidth reinvestment: stream widens, compute fixed; the '
                'wider tied embedding is VRAM, excluded from body accounting'})
    E.train_arm('qk_e15_c', JP, 'E15c', lambda: make_e15c(s=s_c), GC15,
                lr=E7R.muon_lr(),
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'slot': s_c,
                       'design': 'true-small decoders + slot-size '
                                 'reinvestment (communication bandwidth)'})
    E.oldheld_record('qk_e15_c', lambda: make_e15c(s=s_c), JP, 'E15c_oldheld')
    E.paired_fresh('qk_e15_c', JP, 'E15c')
    pair_extra('qk_e15_c', 'E15c', (('qk_e9_a', 'e9a'),
                                    ('qk_e15_b', 'e15b')))
    if not E.SMOKE and 'step_time_E15c' not in E.loadj(JP):
        E.merge(JP, 'step_time_E15c', {
            'sec_per_step_batch16': measure_step_time(
                lambda: make_e15c(s=s_c))})
    print('e15 reinvest run done', flush=True)
