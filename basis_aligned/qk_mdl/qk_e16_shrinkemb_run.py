"""E16 SHRINKING EMBEDDING CHANNEL (Logan 2026-08-05 approved; fresh
single-epoch batch-16, recipe conventions -- per-slot norm + Muon 0.02
(embedding AdamW at the family lr) + in-loss lasso 3e-5 -- paired vs
qk_e9_a AND qk_e0a_vanilla264/qk_e0b on the fresh held set).

CONCEPT: the token embedding is NOT added onto the full stream at the
bottom. Instead, at block i the trailing 264 - 22*i dims (exactly the slots
of the modules that have not yet written; block i contains modules 2i and
2i+1, slot j is owned by module j) carry a TOKEN REMNANT computed directly
from the 264-dim token embedding by a per-block linear W_i: 264 ->
(264 - 22*i), i = 1..11. Block 0 gets the embedding itself in all 264 dims
(identity, no matrix). The remnant REPLACES the content of unclaimed slots
at each block boundary (it is never accumulated through the residual --
every entry is recomputed as [claimed module writes] + [fresh remnant]);
module writes are untouched. After block 11's modules write there is no
block 12, so the READOUT SEES PURE MODULE OUTPUTS (no embedding term).
Per-slot RMSNorm applies to remnant-occupied slots too (the remnant is
slot-aligned; two slots are reclaimed per block).

REMNANT BASE (documented design choice): W_i is applied to the GLOBALLY
RMS-NORMED embedding e = rms_norm(wte(idx)) -- the exact vector the E9a
reference injects -- not the un-normed wte rows. This is a pure per-token
rescaling of the raw embedding (no stream content enters it) and it is what
makes the identity control an EXACT reduction to the E9a-architecture
forward. W_i init = identity slice (selector of the trailing 264-22*i
coords of e; no RNG consumed, base weights stay seed-identical to E9a).

EXTRA PARAMS: 264*(242+220+...+22) = 383,328 (asserted in controls).

ARMS
  E16a  the schedule above, full readable recipe.
  E16b  FLOOR VARIANT (runs after E16a lands, gated on non-divergence):
        the remnant never shrinks below 44 dims = 4 slots. Schedule
        max(264-22i, 44) for i = 1..11 plus a 44-dim remnant AT THE READOUT
        (the floor keeps the i=12 remnant, which the shrink schedule would
        take to 0, at 44): the last 3 consumers (block 10, block 11,
        readout) hold at 44. DISJOINTNESS EXCEPTION (documented in the
        JSON): the trailing 44 dims are slots 20-23, the last 4 modules'
        slots -- at block 11's entry the remnant overlaps the already-
        claimed slots 20/21 (modules 20/21's writes get the remnant ADDED
        on top there), and at the readout it overlaps slots 20-23 (modules
        20-23). Everywhere else remnant and writes stay disjoint.

POSITIVE CONTROLS (before training)
  1. Identity control: mode='control' (no-shrink -- remnant = full-width e
     at EVERY entry including the readout, W matrices unused, module writes
     ENABLED) must equal the E9a-architecture reference (E7R.make_e7m1)
     forward exactly at init: max |logit diff| < 1e-6, fp32, tf32 disabled.
     This is the cleanest exact-identity reduction: with a full-width
     identity remnant, entry(li) = e + sum(writes) = the reference entry.
  2. Param-count assertion: nominal extra == 383,328 (E16a, non-smoke),
     and extra == 264 * sum(schedule) generally.
  3. Lasso penalty (plain V8T.group_penalty on the full-width reads) vs a
     naive per-group loop on the new architecture: rel < 1e-6.
  Plus sanity: the real shrink arm differs from the reference (> 1e-6),
  and structural checks on both schedules.

DIAGNOSTICS in qk_e16.json: per-block token-recovery probe on the remnant
(ridge from the remnant dims to the normed embedding, argmax over the
vocab, fit 48 / eval 16 held seqs -- the E12 funnel-neck style; the remnant
is a per-token function so it is read off wte + W_i on held rows), remnant
norms per block, custom wiring light probe (weight-vs-causal Spearman; the
stream-0 ablation mean is the PER-CONSUMER remnant mean, not the embedding
mean), standard token-determined probes, train/held curves per the brief.
Results -> qk_e16.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, R2, C, DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e16.json')
GC16 = 3e-5
FLOOR_SLOTS = 4                                  # E16b: 4 slots = 44 dims


def rem_schedule(Dm, S, floor_slots=0, depth=DEPTH):
    """Remnant width per consumer li = 0..depth (depth = the readout row).
    li=0: full Dm (identity, no matrix). li>=1: max(Dm - 2*S*li, floor).
    Shrink schedule (floor 0): the readout row is 0 -- no embedding remains.
    Floor schedule: the floor also keeps the readout remnant alive."""
    floor = floor_slots * S
    return [Dm] + [max(Dm - 2 * S * li, floor) for li in range(1, depth + 1)]


class E16Route(E1R.E1Route):
    """Readable recipe WITHOUT bottom embedding injection: consumer li's
    entry = sum of (masked, slot-partitioned) module writes + a token
    remnant W_li(e) scattered into the TRAILING rem_dims[li] dims.
    mode: 'shrink' (E16a), 'floor' (E16b), 'control' (no-shrink identity
    reduction: remnant = e full-width at every entry incl. readout)."""

    def __init__(self, variant, depth, mode='shrink'):
        super().__init__(variant, depth)
        self.mode = mode
        Dm = self.wte.weight.shape[1]
        S = Dm // (2 * depth)
        self.slot = S
        self.rem_dims = rem_schedule(
            Dm, S, FLOOR_SLOTS if mode == 'floor' else 0, depth)
        if mode == 'control':
            self.W_rem = None
        else:
            self.W_rem = nn.ModuleDict()
            with torch.no_grad():
                for li in range(1, depth + 1):
                    r = self.rem_dims[li]
                    if r == 0:
                        continue
                    lin = nn.Linear(Dm, r, bias=False)
                    w = torch.zeros(r, Dm)
                    w[:, Dm - r:] = torch.eye(r)  # identity-slice init, no RNG
                    lin.weight.copy_(w)
                    self.W_rem[str(li)] = lin

    def remnants(self, e):
        """Full-width remnant tensor per consumer (None where width 0)."""
        Dm = e.shape[-1]
        if self.mode == 'control':
            return [e] * (self.depth + 1)
        out = [e]
        for li in range(1, self.depth + 1):
            r = self.rem_dims[li]
            if r == 0:
                out.append(None)
                continue
            Rt = torch.zeros(*e.shape[:-1], Dm, device=e.device, dtype=e.dtype)
            Rt[..., Dm - r:] = self.W_rem[str(li)](e)
            out.append(Rt)
        return out

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))     # remnant base (see docstring)
        rem = self.remnants(e)
        streams = [None]                          # stream 0 is per-consumer
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            tot = sub[0] if (sub is not None and 0 in sub) else rem[li]
            for i in range(1, 1 + 2 * min(li, self.depth)):
                v = sub[i] if (sub is not None and i in sub) else streams[i]
                tot = v if tot is None else tot + v
            return tot

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)                # per-slot (remnant slots too)

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
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)            # per-slot
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)                     # E16a: writes only, no remnant
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))                  # pre-readout norm: global
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e16(variant='E16a', mode='shrink'):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    m = E16Route(variant, DEPTH, mode=mode).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


def make_e16a():
    return make_e16('E16a', 'shrink')


def make_e16b():
    return make_e16('E16b', 'floor')


# ---------------- accounting ----------------
@torch.no_grad()
def body_params(m):
    """Nominal and effective body params (masked write-projection rows
    counted as absent, E15a convention); embedding excluded."""
    tot = sum(p.numel() for p in m.parameters())
    body = tot - m.wte.weight.numel()
    waste = 0
    for k in range(2 * DEPTH):
        blk = m.h[k // 2]
        W = blk.c_proj.weight if k % 2 == 0 else blk.Down.weight
        dead = int((m.wmask[k] == 0).sum())
        waste += dead * W.shape[1]
        if k % 2 == 1:
            waste += dead                         # Down_bias dead entries
    return body, body - waste


def extra_params(m):
    return 0 if m.W_rem is None else \
        sum(p.numel() for p in m.W_rem.parameters())


# ---------------- controls ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    Dm, S = Q.D, E.SUB
    # 1. identity control: no-shrink remnant == E9a-architecture forward
    ref = E7R.make_e7m1().eval().float()
    mc = make_e16('E16ctl', mode='control').eval().float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out_ref = ref(idx)
        d = (mc(idx) - out_ref).abs().max().item()
        ma = make_e16a().eval().float()
        d2 = (ma(idx) - out_ref).abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"control E16(no-shrink: full-width remnant at every entry incl "
          f"readout, writes enabled)==E9a-arch at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-6
    print(f"sanity E16a (shrink schedule) differs from the reference: "
          f"max diff {d2:.2e}", flush=True)
    assert d2 > 1e-6
    # 2. param-count assertions
    extra = extra_params(ma)
    expect = Dm * sum(ma.rem_dims[1:])
    assert extra == expect, (extra, expect)
    if not E.SMOKE:
        assert extra == 383328, extra
        assert ma.rem_dims == [264] + [264 - 22 * i for i in range(1, 12)] \
            + [0], ma.rem_dims
    assert ma.rem_dims[-1] == 0                   # readout sees pure writes
    print(f"control E16a extra remnant params {extra} "
          f"(= {Dm} * sum{ma.rem_dims[1:]})", flush=True)
    # 3. lasso penalty vs naive loop on the new architecture
    p_fast = float(V8T.group_penalty(ma))
    p_naive = 0.0
    for blk in ma.h:
        for nm in E.READ_NAMES:
            M = getattr(blk, nm).weight
            for k in range(E.NGROUP):
                p_naive += float(
                    M[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E16a penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    # E16b structural checks (floor schedule + overlap)
    mb = make_e16b().eval().float()
    fl = FLOOR_SLOTS * S
    assert mb.rem_dims[0] == Dm
    assert all(mb.rem_dims[li] == max(Dm - 2 * S * li, fl)
               for li in range(1, DEPTH + 1)), mb.rem_dims
    assert mb.rem_dims[DEPTH] == fl               # readout remnant held at floor
    if not E.SMOKE:
        assert mb.rem_dims[10] == 44 and mb.rem_dims[11] == 44 \
            and mb.rem_dims[12] == 44
    assert torch.isfinite(mb(idx)).all()
    print(f"control E16b structural: schedule {mb.rem_dims} (floor {fl} dims "
          f"= last {FLOOR_SLOTS} slots; overlap with claimed slots at "
          f"consumers 11 and readout), forward finite", flush=True)
    del ref, mc, ma, mb, out_ref
    torch.cuda.empty_cache()


# ---------------- diagnostics ----------------
N_FIT, N_EV = 48, 16


@torch.no_grad()
def _ridge_top1(H_fit, ids_fit, H_ev, ids_ev, wte_normed):
    """E12 funnel-neck convention: ridge to the normed embedding, argmax."""
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
def remnant_probe(m):
    """Per-consumer token recovery from the remnant + remnant norms. The
    remnant is a per-token function of the embedding, so it is read off
    wte + W_li on held rows directly (no forward needed)."""
    m.eval().float()
    Dm = m.wte.weight.shape[1]
    wten = F.rms_norm(m.wte.weight.detach().float(), (Dm,)).cpu()
    b = Q.HELD[:N_FIT + N_EV, :Q.T]
    ids = b.reshape(-1).cpu()
    nfit = N_FIT * Q.T
    e = F.rms_norm(m.wte(b), (Dm,)).float()
    rec, norms = {}, {}
    for li in range(m.depth + 1):
        name = 'readout' if li == m.depth else f'block{li}'
        if li == 0:
            H = e
        elif m.W_rem is not None and str(li) in m.W_rem:
            H = m.W_rem[str(li)](e)
        else:
            continue
        Hf = H.reshape(-1, H.shape[-1]).cpu()
        acc = _ridge_top1(Hf[:nfit], ids[:nfit], Hf[nfit:], ids[nfit:], wten)
        rec[name] = {'dims': int(H.shape[-1]), 'top1_token_recovery':
                     round(acc, 4)}
        norms[name] = {
            'mean_l2': round(float(Hf.norm(dim=-1).mean()), 4),
            'rms_per_dim': round(float(
                Hf.pow(2).mean().sqrt()), 4)}
        print(f"  remnant probe {name} ({H.shape[-1]} dims): top-1 "
              f"{acc:.4f}, mean l2 {norms[name]['mean_l2']}", flush=True)
    return {'token_recovery': rec, 'remnant_norms': norms,
            'note': 'ridge from the remnant dims to the normed embedding, '
                    'argmax over vocab; fit 48 held seqs, eval 16 (E12 '
                    'funnel-neck style); remnant computed from wte + W_i '
                    '(per-token function, no context)'}


@torch.no_grad()
def e16_light_probe(m):
    """E.light_probe with the stream-0 correction: the ablation mean for the
    token channel is the PER-CONSUMER remnant mean (E.light_probe's
    base_and_means hardcodes stream 0 = full normed embedding)."""
    m.eval().float()
    base, means = R2.base_and_means(m)            # means[0] unused (wrong here)
    sums, n = None, 0
    for i in range(0, R2.ABL_N, 8):
        b = R2.HELD[i:i + 8]
        e = F.rms_norm(m.wte(b[:, :R2.T]), (Q.D,))
        rems = m.remnants(e)
        if sums is None:
            sums = [None if r is None else
                    torch.zeros(Q.D, dtype=torch.float64, device=e.device)
                    for r in rems]
        for li, r in enumerate(rems):
            if r is not None:
                sums[li] += r.double().sum((0, 1))
        n += b.shape[0] * R2.T
    rem_means = [None if s is None else (s / n).float() for s in sums]
    dce = {}
    for li in range(DEPTH + 1):
        dce[li] = {}
        for si in m.vis[li]:
            if si == 0:
                if rem_means[li] is None:
                    continue                      # no remnant at this consumer
                mv = rem_means[li]
            else:
                mv = means[si]
            dce[li][si] = R2.ce_with(m, sub_entry={li: {si: mv}}) - base
        top = sorted(dce[li], key=lambda s: -dce[li][s])[:4]
        print(f"  consumption row {li}: " + ", ".join(
            f"{R2.stream_name(si)} {dce[li][si]:+.4f}" for si in top),
            flush=True)
    scores = E.weight_support(m)
    wp = [(li, si) for li in scores for si in scores[li]]
    sup = [scores[li][si] for li, si in wp]
    cau = [dce[li][si] for li, si in wp]
    eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    totals = {}
    for j in range(DEPTH):
        totals[str(j)] = round(sum(dce[li].get(si, 0.0) for li in dce
                                   for si in (1 + 2 * j, 2 + 2 * j)
                                   if si in dce[li]), 5)
    pairs = sorted([(li, si, dce[li][si]) for li in dce for si in dce[li]],
                   key=lambda p: -p[2])
    r = {'base_ce_fp32_abl_oldheld': round(base, 5),
         'wiring_n_pairs': len(wp),
         'wiring_spearman_all': round(R2.spearman(sup, cau), 4),
         'wiring_n_effectual': len(eff),
         'wiring_spearman_effectual': round(R2.spearman(
             [sup[k] for k in eff], [cau[k] for k in eff]), 4),
         'wiring_top10_precision': len(top_sup & top_cau) / 10.0,
         'per_block_total_consumption': totals,
         'dead_blocks_below_0.001': [j for j in totals if totals[j] < 0.001],
         'consumption_top20': [
             {'consumer': ('readout' if li == DEPTH else f'block{li}'),
              'source': R2.stream_name(si), 'dce': round(v, 5)}
             for li, si, v in pairs[:20]],
         'consumption_matrix': {str(li): {str(si): round(v, 6)
                                          for si, v in dce[li].items()}
                                for li in dce},
         'weight_support_matrix': {str(li): {str(si): round(v, 3)
                                             for si, v in scores[li].items()}
                                   for li in scores},
         'note': 'stream 0 = per-consumer token remnant; its ablation mean '
                 'is the per-consumer remnant mean on the cooc ABL rows'}
    print(f"light probe: Spearman all {r['wiring_spearman_all']} "
          f"effectual({len(eff)}) {r['wiring_spearman_effectual']} "
          f"top10 {r['wiring_top10_precision']}", flush=True)
    return r


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') \
                and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def run_arm(stem, key, factory, design, extra_pairs=()):
    mlr = E7R.muon_lr()
    E.train_arm(stem, JP, key, factory, GC16, lr=mlr,
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'design': design})
    E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
    E.paired_fresh(stem, JP, key)
    pair_extra(stem, key, (('qk_e9_a', 'e9a'),) + tuple(extra_pairs))
    if not E.SMOKE and os.path.exists(E.ckpath(stem)):
        m = None

        def need(k):
            return k not in E.loadj(JP)
        if need(f'remnant_probe_{key}'):
            m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'remnant_probe_{key}', remnant_probe(m))
        if need(f'light_probe_{key}'):
            if m is None:
                m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'light_probe_{key}', e16_light_probe(m))
        if need(f'tok_probe_{key}'):
            if m is None:
                m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'tok_probe_{key}', E.tok_probe(m))
        if m is not None:
            del m
            torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()

    if 'E16_config' not in E.loadj(JP):
        ma = make_e16a()
        mb = make_e16b()
        nom_a, eff_a = body_params(ma)
        nom_b, eff_b = body_params(mb)
        E.merge(JP, 'E16_config', {
            'stream_width': Q.D, 'slots': E.NGROUP, 'slot_dim': E.SUB,
            'remnant_schedule_E16a': ma.rem_dims,
            'remnant_schedule_E16b': mb.rem_dims,
            'extra_remnant_params_E16a': extra_params(ma),
            'extra_remnant_params_E16b': extra_params(mb),
            'nominal_body_E16a': nom_a, 'effective_body_E16a': eff_a,
            'nominal_body_E16b': nom_b, 'effective_body_E16b': eff_b,
            'e9a_reference_body': 'nominal 15,057,504 / effective '
                                  '11,049,984 (qk_e15.json E15a)',
            'remnant_base': 'globally RMS-normed embedding e = '
                            'rms_norm(wte(idx)) -- the exact vector the E9a '
                            'reference injects; makes the identity control '
                            'an exact reduction (per-slot norm rescales '
                            'remnant slots at module inputs regardless)',
            'w_init': 'identity slice: W_i starts as the selector of the '
                      'trailing rem_dims[i] coords of e (no RNG consumed)',
            'optimizer_note': 'W_i are 2D body matrices -> Muon at the '
                              'family muon lr; embedding stays on AdamW',
            'E16b_disjointness_exception':
                'floor = 44 dims = slots 20-23 (the last 4 modules\' '
                'slots). Consumers 10/11/readout hold at 44. At block '
                '11\'s entry the remnant overlaps the already-claimed '
                'slots 20/21 (modules 20/21\'s writes get the remnant '
                'ADDED on top); at the readout it overlaps slots 20-23 '
                '(modules 20-23). All other remnant placements occupy '
                'only unclaimed slots (pure replacement, disjoint from '
                'every write).',
            'E16b_schedule_interpretation':
                '"never below 44" applied to consumers i=1..11 AND the '
                'readout row (which the shrink schedule takes to 0): this '
                'is the reading under which "last 3 blocks hold at 44" '
                '(10, 11, readout) and "overlap with the last 4 modules\' '
                'slots" are both exact.'})
        del ma, mb
        torch.cuda.empty_cache()

    run_arm('qk_e16_a', 'E16a', make_e16a,
            'shrinking embedding channel: no bottom injection; per-block '
            'W_i: 264 -> 264-22i token remnant in the unclaimed trailing '
            'slots, replaced (not accumulated) at each block boundary; '
            'readout sees pure module outputs; full readable recipe')

    rec_a = E.loadj(JP).get('E16a', {})
    if rec_a.get('diverged'):
        E.merge(JP, 'E16b_skipped', 'gate: E16a diverged')
    else:
        run_arm('qk_e16_b', 'E16b', make_e16b,
                'floor variant: remnant never shrinks below 44 dims '
                '(4 slots); consumers 10/11/readout hold at 44; overlap '
                'with the last 4 modules\' slots documented as the '
                'disjointness exception',
                extra_pairs=(('qk_e16_a', 'e16a'),))
    print('e16 shrinkemb run done', flush=True)
