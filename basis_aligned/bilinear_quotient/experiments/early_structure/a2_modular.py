"""A2 — modular addition: ∗-algebra blocks and grokking.

DGP. x = [onehot(a) | onehot(b)] ∈ R^{2p}, target class (a+b) mod p, cross-entropy,
a random train/test split of the p² pairs, weight decay (grokking).

Known ground truth. logit_c = Σ_ω α_ω cos(2πω(a+b-c)/p), and
cos(θ_a+θ_b-θ_c) = [cosθ_a cosθ_b - sinθ_a sinθ_b]cosθ_c
                 + [sinθ_a cosθ_b + cosθ_a sinθ_b]sinθ_c,
so in the Fourier basis Q_c is block diagonal with one 4-dim block per frequency
(spanned by cos_ω^a, sin_ω^a, cos_ω^b, sin_ω^b), and inside a block the a↔b
cross part is the reflection [[cosθ_c, sinθ_c],[sinθ_c, -cosθ_c]] · α_ω/2.

Claims tested (doc A2):
 (i)   the retained {Q_c} simultaneously block-diagonalise; the commutant (Maehara–
       Murota) recovers the frequency blocks from weights alone, no probing;
 (ii)  the block structure survives the gauge-scramble null;
 (iii) block crystallisation tracks / precedes the generalisation transition.
Verification: ablate a recovered block ⇒ that frequency's logit contribution
vanishes; splice blocks between independently trained models ⇒ function transfers.
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, sbd, sbd_robust, block_mass,
                       support_basis, restrict, isotypic_groups, gauge_refactor,
                       lam_cos, lam_relerr, effective_rank, forward_Q)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

P = 23
H = 128
STEPS = 40000
CKPT_EVERY = 500
TRAIN_FRAC = 0.4
WD = 1.0
LR = 1e-3


# ------------------------------------------------------------------ data / basis

def all_pairs():
    a = torch.arange(P, device=DEV).repeat_interleave(P)
    b = torch.arange(P, device=DEV).repeat(P)
    x = torch.zeros(P * P, 2 * P, device=DEV)
    x[torch.arange(P * P), a] = 1.0
    x[torch.arange(P * P), P + b] = 1.0
    return x, (a + b) % P, a, b


def split(seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(P * P, generator=g).to(DEV)
    n_tr = int(TRAIN_FRAC * P * P)
    return perm[:n_tr], perm[n_tr:]


def fourier_blocks():
    """For each frequency ω=1..(p-1)/2, the orthonormal 4-dim planted subspace
    [cos_ω^a, sin_ω^a, cos_ω^b, sin_ω^b] of R^{2p}. Plus the 2-dim DC block."""
    t = torch.arange(P, device=DEV, dtype=torch.get_default_dtype())
    blocks = {}
    dc = torch.zeros(2 * P, 2, device=DEV)
    dc[:P, 0] = 1 / math.sqrt(P)
    dc[P:, 1] = 1 / math.sqrt(P)
    blocks[0] = dc
    for w in range(1, P // 2 + 1):
        c = torch.cos(2 * math.pi * w * t / P) * math.sqrt(2 / P)
        s = torch.sin(2 * math.pi * w * t / P) * math.sqrt(2 / P)
        B = torch.zeros(2 * P, 4, device=DEV)
        B[:P, 0], B[:P, 1] = c, s
        B[P:, 2], B[P:, 3] = c, s
        blocks[w] = B
    return blocks


FBLOCKS = fourier_blocks()


# ------------------------------------------------------------------ training

@torch.no_grad()
def acc_loss(p, x, y, idx):
    out = forward(p, x[idx])
    return (float((out.argmax(1) == y[idx]).to(torch.get_default_dtype()).mean()),
            float(F.cross_entropy(out, y[idx])))


def train_model(seed, steps=STEPS, wd=WD, shuffle_labels=False, log=True):
    x, y, a, b = all_pairs()
    tr, te = split(seed)
    if shuffle_labels:
        g = torch.Generator().manual_seed(seed + 7)
        y = y.clone()
        y[tr] = y[tr][torch.randperm(len(tr), generator=g).to(DEV)]
    p = init_params(2 * P, H, P, seed=seed, device=DEV, scale=1.0)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=LR, weight_decay=wd)
    hist, ckpts = [], []
    for s in range(steps + 1):
        loss = F.cross_entropy(forward(p, x[tr]), y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % CKPT_EVERY == 0 or s == steps:
            tra, trl = acc_loss(p, x, y, tr)
            tea, tel = acc_loss(p, x, y, te)
            hist.append({'step': s, 'train_acc': tra, 'test_acc': tea,
                         'train_loss': trl, 'test_loss': tel})
            ckpts.append({k: v.detach().clone() for k, v in p.items()})
            if log and s % (CKPT_EVERY * 10) == 0:
                print(f'    step {s:6d} train {tra:.3f}/{trl:.3e}  test {tea:.3f}/{tel:.3e}')
    for k in p:
        p[k].requires_grad_(False)
    return p, hist, ckpts


# ------------------------------------------------------------------ analysis

def fourier_power(Q):
    """Fraction of ‖Q‖²_F carried by each planted frequency block (as a
    block-diagonal projection), plus the leftover off-block mass."""
    tot = float((Q ** 2).sum())
    out = {}
    covered = 0.0
    for w, B in FBLOCKS.items():
        Pb = B @ B.T
        e = float((torch.einsum('ij,mjk,kl->mil', Pb, Q, Pb) ** 2).sum())
        out[w] = e / tot
        covered += e
    out['off_block'] = 1 - covered / tot
    return out


def sbd_analysis(Q, thresh=1e-3, seed=0):
    """Weights-only block recovery: restrict to the joint range, SBD there, then
    ask what each recovered block IS (overlap with planted Fourier subspaces)."""
    U, sv = support_basis(Q, thresh=thresh)
    k = U.shape[1]
    if k < 2:
        return {'support_dim': k, 'blocks': [], 'in_block_mass': float('nan')}
    Qr = restrict(Q, U)
    Pm, sizes, info = sbd(Qr, gap_rel=1e-5, seed=seed)
    mass, T = block_mass(Qr, Pm, sizes)
    # identify each recovered block with a planted frequency
    Vfull = U @ Pm                       # (d, k) recovered directions in input space
    blocks, s0 = [], 0
    for sz in sizes:
        Vb = Vfull[:, s0:s0 + sz]
        best_w, best_ov = None, -1.0
        for w, B in FBLOCKS.items():
            ov = float(((Vb.T @ B) ** 2).sum() / sz)     # captured energy in [0,1]
            if ov > best_ov:
                best_w, best_ov = w, ov
        blocks.append({'size': sz, 'freq': best_w, 'overlap': best_ov,
                       'mass': float((T[:, s0:s0 + sz, s0:s0 + sz] ** 2).sum() / (T ** 2).sum())})
        s0 += sz
    return {'support_dim': k, 'commutant_dim': info['commutant_dim'],
            'sizes': sizes, 'in_block_mass': mass, 'blocks': blocks,
            'n_size4': sum(1 for s in sizes if s == 4),
            'mean_overlap_size4': float(sum(b['overlap'] for b in blocks if b['size'] == 4) /
                                        max(1, sum(1 for s in sizes if s == 4))),
            'support_sv': [float(v) for v in sv[:16]]}


def phase_coherence(Q, w):
    """Reader-indexed claim: inside frequency ω's planted block the per-class 2×2
    cross matrix must be the reflection at angle θ_c = 2πωc/p. Returns the
    circular R² of the recovered angle against the predicted line. Destroyed by
    the reader-shuffle null; invariant to gauge."""
    B = FBLOCKS[w]
    Qb = torch.einsum('ip,mij,jq->mpq', B, Q, B)          # (p,4,4)
    M = Qb[:, :2, 2:]                                     # a-side x b-side
    ang = torch.atan2(M[:, 0, 1] + M[:, 1, 0], M[:, 0, 0] - M[:, 1, 1])
    c = torch.arange(P, device=Q.device, dtype=ang.dtype)
    pred = 2 * math.pi * w * c / P
    r = torch.exp(1j * (ang - pred)).mean().abs()         # circular concentration
    return float(r), [float(v) for v in ang]


@torch.no_grad()
def block_ablation(Q, V, x, y, idx):
    """Project the family off the subspace V (d,k) and re-evaluate."""
    Pk = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype) - V @ V.T
    Qa = torch.einsum('ij,mjk,kl->mil', Pk, Q, Pk)
    out = forward_Q(Qa, x[idx])
    return (float((out.argmax(1) == y[idx]).to(Q.dtype).mean()),
            float(F.cross_entropy(out, y[idx])), Qa)


@torch.no_grad()
def freq_logit_contribution(Q, x, w):
    """Variance of the logit contribution of planted frequency ω, and its
    correlation with the ideal cos(2πω(a+b-c)/p) pattern."""
    B = FBLOCKS[w]
    Pb = B @ B.T
    Qw = torch.einsum('ij,mjk,kl->mil', Pb, Q, Pb)
    yw = forward_Q(Qw, x)                                  # (n, p)
    _, _, a, b = all_pairs()
    c = torch.arange(P, device=x.device, dtype=x.dtype)
    ideal = torch.cos(2 * math.pi * w * ((a + b)[:, None] - c[None, :]) / P)
    yc = yw - yw.mean(1, keepdim=True)
    ic = ideal - ideal.mean(1, keepdim=True)
    corr = float((yc * ic).sum() / (yc.norm() * ic.norm()).clamp_min(1e-30))
    return {'var': float(yw.var()), 'corr_ideal': corr}


# ------------------------------------------------------------ canonicalisation

def identifiable_basis(x):
    """Orthonormal basis of span{vec(xxᵀ) : x in the data} inside Sym²(V).

    One-hot inputs probe only a 529-dimensional slice of the 1081-dimensional
    Sym², so most of a learned Q is not identifiable from the function at all.
    Any weight-space geometry claim has to be made about the identifiable part —
    the rest is a gauge of the data, not of the model.
    """
    d = x.shape[1]
    iu, ju = torch.triu_indices(d, d)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(x)
    Phi = (x[:, iu] * x[:, ju]) * sc
    _, s, Vh = torch.linalg.svd(Phi, full_matrices=False)
    r = int((s > 1e-8 * s.max()).sum())
    return Vh[:r], iu, ju, sc


def canonicalise(Q, basis):
    """Minimum-norm representative of the same function on the data manifold."""
    Vh, iu, ju, sc = basis
    V = Q[:, iu, ju] * sc
    Vid = (V @ Vh.T) @ Vh
    out = torch.zeros_like(Q)
    out[:, iu, ju] = Vid / sc
    out[:, ju, iu] = Vid / sc
    return out, float((Vid ** 2).sum() / (V ** 2).sum())


def block_project(Q, ws):
    out = torch.zeros_like(Q)
    for w in ws:
        Pb = FBLOCKS[w] @ FBLOCKS[w].T
        out = out + torch.einsum('ij,mjk,kl->mil', Pb, Q, Pb)
    return out


@torch.no_grad()
def functional(Q, x, y, idx=None):
    """Centred logits (CE only sees differences across classes) and their scores."""
    o = forward_Q(Q, x)
    o = o - o.mean(1, keepdim=True)
    sel = slice(None) if idx is None else idx
    return {'acc': float((o[sel].argmax(1) == y[sel]).to(Q.dtype).mean()),
            'ce': float(F.cross_entropy(o[sel], y[sel])), 'logits': o}


def functional_residual(Qa, Qb, x):
    """‖y_a - y_b‖² / ‖y_a‖² on centred logits over the data manifold."""
    oa = forward_Q(Qa, x)
    oa = oa - oa.mean(1, keepdim=True)
    ob = forward_Q(Qb, x)
    ob = ob - ob.mean(1, keepdim=True)
    return float(((oa - ob) ** 2).sum() / (oa ** 2).sum().clamp_min(1e-30))


# ------------------------------------------------------------------ main

CALIB = {'exact_sbd_max_offblock': 1e-5, 'robust_auto_max_offblock': 0.0099,
         'robust_hinted_max_offblock': 0.039}


def sbd_report(Q, seed=0):
    """Weights-only block recovery, with the calibrated verdict attached."""
    U, sv = support_basis(Q, thresh=1e-3)
    cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    k = int((cum < 0.999).sum()) + 1
    Ur = U[:, :k]
    Qr = restrict(Q, Ur)
    rep = {'support_dim': int(U.shape[1]), 'support_dim_kept': k}
    Pe, se, ie = sbd(Qr, gap_rel=1e-5)
    rep['exact'] = {'n_blocks': len(se), 'sizes': se[:24],
                    'commutant_dim': ie['commutant_dim']}
    Pr, sr, ir = sbd_robust(Qr, seed=seed)
    gr = isotypic_groups(Qr, Pr, sr, commutant_dim=ir['commutant_dim'])
    V = Ur @ Pr
    offs = [0]
    for s_ in sr:
        offs.append(offs[-1] + s_)
    comps = []
    for g in gr:
        cols = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in g])
        Vb = V[:, cols]
        w = max(FBLOCKS, key=lambda kk: float(((Vb.T @ FBLOCKS[kk]) ** 2).sum()))
        comps.append({'dim': int(Vb.shape[1]), 'freq': int(w),
                      'overlap': float(((Vb.T @ FBLOCKS[w]) ** 2).sum() / Vb.shape[1])})
    rep['robust'] = {'n_fine_blocks': len(sr), 'n_components': len(gr),
                     'in_block_mass': ir['in_block_mass'], 'defect': ir['defect'],
                     'commutant_dim': ir['commutant_dim'],
                     'n_freq_recovered': sum(1 for c in comps if c['dim'] == 4 and c['overlap'] > 0.9),
                     'component_dims': sorted(c['dim'] for c in comps)}
    return rep


def per_model(p, seed, x, y, basis, verbose=True):
    tr, te = split(seed)
    Q = interaction(p)
    Qid, idfrac = canonicalise(Q, basis)
    ws = list(range(0, P // 2 + 1))
    Qb = block_project(Qid, ws)
    fp_raw, fp_id = fourier_power(Q), fourier_power(Qid)
    full, blocked = functional(Qid, x, y), functional(Qb, x, y)
    r = {
        'seed': seed,
        'test_acc': float((full['logits'][te].argmax(1) == y[te]).to(Q.dtype).mean()),
        'identifiable_mass_frac': idfrac,
        'off_block_raw': fp_raw['off_block'], 'off_block_canonical': fp_id['off_block'],
        'functional_residual_of_block_projection': functional_residual(Qid, Qb, x),
        'freq_power': {int(k): round(v, 5) for k, v in fp_id.items() if k != 'off_block'},
        'surgery': {
            'test_acc_before': float((full['logits'][te].argmax(1) == y[te]).to(Q.dtype).mean()),
            'test_acc_after': float((blocked['logits'][te].argmax(1) == y[te]).to(Q.dtype).mean()),
            'test_ce_before': float(F.cross_entropy(full['logits'][te], y[te])),
            'test_ce_after': float(F.cross_entropy(blocked['logits'][te], y[te])),
            'train_acc_after': float((blocked['logits'][tr].argmax(1) == y[tr]).to(Q.dtype).mean())},
        'sbd': sbd_report(Qid, seed=seed),
        'phase_coherence': {w: round(phase_coherence(Qid, w)[0], 4) for w in range(1, P // 2 + 1)},
    }
    r['sbd']['verdict'] = ('recoverable' if r['off_block_canonical'] <= CALIB['robust_hinted_max_offblock']
                           else 'TOOL-LIMITED (off-block beyond calibrated range)')
    if verbose:
        s = r['surgery']
        print(f"  seed {seed}: test {r['test_acc']:.4f} | identifiable {idfrac:.3f} | off-block "
              f"{r['off_block_raw']:.4f} raw / {r['off_block_canonical']:.4f} canonical | "
              f"block-projection: functional residual {r['functional_residual_of_block_projection']:.4f}, "
              f"test acc {s['test_acc_before']:.4f} -> {s['test_acc_after']:.4f}, CE "
              f"{s['test_ce_before']:.4f} -> {s['test_ce_after']:.4f}")
        print(f"    SBD: exact {r['sbd']['exact']['n_blocks']} blocks | robust "
              f"{r['sbd']['robust']['n_components']} components, {r['sbd']['robust']['n_freq_recovered']}/11 "
              f"frequencies, defect {r['sbd']['robust']['defect']:.3f} -> {r['sbd']['verdict']}")
        print(f"    phase coherence (should be ~1 for used frequencies): "
              f"{[r['phase_coherence'][w] for w in range(1, 6)]} ...")
    return r, Q, Qid, Qb


def main():
    t0 = time.time()
    x, y, a, b = all_pairs()
    basis = identifiable_basis(x)
    out = {'config': {'p': P, 'h': H, 'steps': STEPS, 'train_frac': TRAIN_FRAC,
                      'wd': WD, 'lr': LR, 'device': DEV},
           'calibration_used': CALIB}

    if os.path.exists('a2_cache.pt'):
        cache = torch.load('a2_cache.pt', weights_only=False)
        print('== loaded cached models ==')
    else:
        cache = {}
        for seed in range(3):
            print(f'== train seed {seed} ==')
            p, hist, ck = train_model(seed)
            cache[seed] = {'p': p, 'hist': hist, 'ck': ck}
        torch.save(cache, 'a2_cache.pt')

    print('== per-model analysis ==')
    out['main'] = []
    Qids, Qbs = [], []
    for seed in range(3):
        r, Q, Qid, Qb = per_model(cache[seed]['p'], seed, x, y, basis)
        r['final'] = cache[seed]['hist'][-1]
        out['main'].append(r)
        Qids.append(Qid)
        Qbs.append(Qb)

    print('== grokking dynamics (seed 0): does block structure crystallise? ==')
    dyn = []
    Qfin = interaction(cache[0]['p'])
    tr0, te0 = split(0)
    for i, (rec, ck) in enumerate(zip(cache[0]['hist'], cache[0]['ck'])):
        Q = interaction(ck)
        Qid, idf = canonicalise(Q, basis)
        Qb = block_project(Qid, list(range(0, P // 2 + 1)))
        ob = functional(Qb, x, y)
        d_ = {**rec, 'identifiable_mass_frac': idf,
              'off_block_raw': fourier_power(Q)['off_block'],
              'off_block_canonical': fourier_power(Qid)['off_block'],
              'functional_residual_block': functional_residual(Qid, Qb, x),
              'blocked_test_acc': float((ob['logits'][te0].argmax(1) == y[te0]).to(Q.dtype).mean()),
              'cos_to_final': lam_cos(Q, Qfin)}
        if i % 10 == 0:
            d_['sbd_defect'] = sbd_report(Qid)['robust']['defect']
        dyn.append(d_)
    out['dynamics'] = dyn
    for r in dyn[::max(1, len(dyn) // 14)]:
        print(f"  step {r['step']:6d} test {r['test_acc']:.3f} | off-block {r['off_block_canonical']:.3f} "
              f"| block-proj test {r['blocked_test_acc']:.3f} | fn residual "
              f"{r['functional_residual_block']:.3f} | cos->final {r['cos_to_final']:.3f}")

    print('== verification: ablate one frequency block ==')
    Qb0 = Qbs[0]
    base = functional(Qb0, x, y)
    abl = []
    for w in range(1, P // 2 + 1):
        keep = [k for k in range(0, P // 2 + 1) if k != w]
        Qa = block_project(Qb0, keep)
        fa = functional(Qa, x, y)
        before = freq_logit_contribution(Qb0, x, w)
        after = freq_logit_contribution(Qa, x, w)
        g = torch.Generator().manual_seed(w)
        Vr = torch.linalg.qr(torch.randn(2 * P, 4, generator=g).to(Qb0))[0]
        Pk = torch.eye(2 * P, device=Qb0.device, dtype=Qb0.dtype) - Vr @ Vr.T
        fr = functional(torch.einsum('ij,mjk,kl->mil', Pk, Qb0, Pk), x, y)
        abl.append({'freq': w, 'acc': fa['acc'], 'ce': fa['ce'],
                    'ctrl_random_4dim_acc': fr['acc'], 'ctrl_random_4dim_ce': fr['ce'],
                    'freq_var_before': before['var'], 'freq_var_after': after['var'],
                    'freq_var_ratio': after['var'] / max(before['var'], 1e-30),
                    'corr_ideal': before['corr_ideal']})
        print(f"  ablate freq {w:2d}: acc {base['acc']:.4f} -> {fa['acc']:.4f} (random-4dim ctrl "
              f"{fr['acc']:.4f}) | that frequency's logit variance x{abl[-1]['freq_var_ratio']:.1e} "
              f"| corr with ideal cos pattern {before['corr_ideal']:.4f}")
    out['ablation'] = {'base': {'acc': base['acc'], 'ce': base['ce']}, 'blocks': abl}

    print('== verification: cross-model block splice ==')
    out['splice'] = splice_experiment([cache[0]['p'], cache[1]['p']], x, y)

    print('== NULL 1: random weights ==')
    out['null_random'] = []
    for seed in range(3):
        p = init_params(2 * P, H, P, seed=1000 + seed, device=DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        Q = interaction(p)
        Qid, idf = canonicalise(Q, basis)
        Qb = block_project(Qid, list(range(0, P // 2 + 1)))
        fb = functional(Qb, x, y)
        o = {'seed': seed, 'identifiable_mass_frac': idf,
             'off_block_canonical': fourier_power(Qid)['off_block'],
             'functional_residual_block': functional_residual(Qid, Qb, x),
             'blocked_acc': fb['acc'],
             'sbd': sbd_report(Qid),
             'phase_coherence': {w: round(phase_coherence(Qid, w)[0], 4) for w in range(1, 6)}}
        print(f"  random {seed}: off-block {o['off_block_canonical']:.4f} fn-residual "
              f"{o['functional_residual_block']:.4f} blocked acc {o['blocked_acc']:.4f} "
              f"phase {list(o['phase_coherence'].values())}")
        out['null_random'].append(o)

    print('== NULL 2: gauge scramble ==')
    out['null_gauge'] = []
    for seed in range(2):
        pg, resid = gauge_refactor(cache[seed]['p'], seed=555 + seed)
        Qg = interaction(pg)
        Qgi, _ = canonicalise(Qg, basis)
        o = {'seed': seed, 'refactor_residual': resid,
             'Q_relerr_vs_trained': lam_relerr(interaction(cache[seed]['p']), Qg),
             'off_block_canonical_trained': out['main'][seed]['off_block_canonical'],
             'off_block_canonical_gauged': fourier_power(Qgi)['off_block'],
             'phase_coherence_trained': out['main'][seed]['phase_coherence'],
             'phase_coherence_gauged': {w: round(phase_coherence(Qgi, w)[0], 4)
                                        for w in range(1, P // 2 + 1)},
             'weight_stat_trained': weight_fourier_stat(cache[seed]['p']),
             'weight_stat_gauged': weight_fourier_stat(pg)}
        print(f"  seed {seed}: residual {resid:.1e} ΔQ {o['Q_relerr_vs_trained']:.1e} | INVARIANT: "
              f"off-block {o['off_block_canonical_trained']:.4f} -> {o['off_block_canonical_gauged']:.4f}"
              f" | DIES: L-row Fourier concentration {o['weight_stat_trained']:.4f} -> "
              f"{o['weight_stat_gauged']:.4f}")
        out['null_gauge'].append(o)

    print('== NULL 3: task shuffle ==')
    out['null_taskshuffle'] = []
    for seed in range(2):
        p, hist, _ = train_model(seed, shuffle_labels=True, log=False)
        Q = interaction(p)
        Qid, idf = canonicalise(Q, basis)
        Qb = block_project(Qid, list(range(0, P // 2 + 1)))
        fb = functional(Qb, x, y)
        tr, te = split(seed)
        o = {'seed': seed, 'train_acc': hist[-1]['train_acc'], 'test_acc': hist[-1]['test_acc'],
             'off_block_canonical': fourier_power(Qid)['off_block'],
             'functional_residual_block': functional_residual(Qid, Qb, x),
             'blocked_test_acc': float((fb['logits'][te].argmax(1) == y[te]).to(Q.dtype).mean()),
             'phase_coherence': {w: round(phase_coherence(Qid, w)[0], 4) for w in range(1, 6)}}
        print(f"  shuffled {seed}: train {o['train_acc']:.3f} test {o['test_acc']:.3f} | off-block "
              f"{o['off_block_canonical']:.4f} | fn-residual {o['functional_residual_block']:.4f} | "
              f"block-proj test {o['blocked_test_acc']:.3f} | phase {list(o['phase_coherence'].values())}")
        out['null_taskshuffle'].append(o)

    print('== NULL 4: reader shuffle ==')
    out['null_readershuffle'] = []
    for seed in range(2):
        Qid = Qids[seed]
        g = torch.Generator().manual_seed(77 + seed)
        Qp = Qid[torch.randperm(P, generator=g).to(Qid.device)]
        pc0 = {w: round(phase_coherence(Qid, w)[0], 4) for w in range(1, P // 2 + 1)}
        pcs = {w: round(phase_coherence(Qp, w)[0], 4) for w in range(1, P // 2 + 1)}
        o = {'seed': seed, 'off_block_true': fourier_power(Qid)['off_block'],
             'off_block_shuffled': fourier_power(Qp)['off_block'],
             'phase_coherence_true': pc0, 'phase_coherence_shuffled': pcs,
             'mean_phase_true': sum(pc0.values()) / len(pc0),
             'mean_phase_shuffled': sum(pcs.values()) / len(pcs)}
        print(f"  seed {seed}: set-level structure UNCHANGED (off-block {o['off_block_true']:.4f} -> "
              f"{o['off_block_shuffled']:.4f}) but reader-indexed phase coherence "
              f"{o['mean_phase_true']:.4f} -> {o['mean_phase_shuffled']:.4f}")
        out['null_readershuffle'].append(o)

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path}  ({out["runtime_s"]:.0f}s)')


def weight_fourier_stat(p):
    """Basis-reading statistic: mean concentration of each L row on its single
    best Fourier frequency (a-side), read in the privileged token basis. High for
    a trained model; must be destroyed by the gauge null."""
    t = torch.arange(P, device=p['L'].device, dtype=p['L'].dtype)
    F_ = torch.stack([torch.cos(2 * math.pi * w * t / P) for w in range(1, P // 2 + 1)] +
                     [torch.sin(2 * math.pi * w * t / P) for w in range(1, P // 2 + 1)])
    F_ = F_ / F_.norm(dim=1, keepdim=True)
    pw = (p['L'][:, :P] @ F_.T) ** 2
    pw = pw[:, :P // 2] + pw[:, P // 2:]
    return float((pw.max(1).values / pw.sum(1).clamp_min(1e-30)).mean())


def splice_experiment(models, x, y):
    """Canonicalise two independently trained models into planted frequency
    blocks, then build a hybrid taking each frequency's block from one model."""
    QA, QB = block_project(interaction(models[0]), list(range(1, P // 2 + 1))), \
        block_project(interaction(models[1]), list(range(1, P // 2 + 1)))
    freqs = list(range(1, P // 2 + 1))
    half = len(freqs) // 2
    sA, sB = freqs[:half], freqs[half:]

    def ev(Q):
        return functional(Q, x, y)['acc']

    hybrid = block_project(QA, sA) + block_project(QB, sB)

    def scaled(Q, ws, ref):
        out = torch.zeros_like(Q)
        for w in ws:
            blk = block_project(Q, [w])
            out = out + blk / blk.norm().clamp_min(1e-30) * ref
        return out

    ref = block_project(QA, [sA[0]]).norm()
    hybrid_sc = scaled(QA, sA, ref) + scaled(QB, sB, ref)
    g = torch.Generator().manual_seed(3)
    ctrl = block_project(QA, sA).clone()
    for w in sB:                     # rotate B's blocks inside the block: breaks phase, keeps energy
        Bm = FBLOCKS[w]
        Rk = torch.linalg.qr(torch.randn(4, 4, generator=g).to(Bm))[0]
        blk = block_project(QB, [w])
        core = torch.einsum('ip,mij,jq->mpq', Bm, blk, Bm)
        Bp = Bm @ Rk
        ctrl = ctrl + torch.einsum('ip,mpq,jq->mij', Bp, core, Bp)
    res = {'freqs_A': sA, 'freqs_B': sB,
           'acc_A_full': ev(QA), 'acc_B_full': ev(QB),
           'acc_A_half': ev(block_project(QA, sA)), 'acc_B_half': ev(block_project(QB, sB)),
           'acc_hybrid': ev(hybrid), 'acc_hybrid_scalematched': ev(hybrid_sc),
           'acc_hybrid_phase_scrambled_ctrl': ev(ctrl)}
    print(f"  A {res['acc_A_full']:.4f} B {res['acc_B_full']:.4f} | halves {res['acc_A_half']:.4f}/"
          f"{res['acc_B_half']:.4f} | HYBRID {res['acc_hybrid']:.4f} (scale-matched "
          f"{res['acc_hybrid_scalematched']:.4f}) | phase-scrambled ctrl "
          f"{res['acc_hybrid_phase_scrambled_ctrl']:.4f}")
    return res


if __name__ == '__main__':
    main()
