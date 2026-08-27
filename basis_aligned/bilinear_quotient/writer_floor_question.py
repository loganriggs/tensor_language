# writer_floor_question: THE LAST UNCONTROLLED CLAIM IN THE WRITER GRAPH.
# S1606 ran the random-subspace control at pronouns@mlp17 and found a generic
# high-norm floor: a RANDOM rank-8 basis returns mlp17/mlp16/mlp15 as its top
# "writers", and floor-corrected, two real slice rules share ZERO components.
# S1608 then showed the floor does NOT reach within-layer head grain (the law
# survived at 3.1-18.8x over random), so the head_grain caveat was lifted.
#
# What remains uncontrolled is S1597's COMPONENT-level headline: question@mlp11
# rank-2 |lambda| slice, top-4 writers {attn10, attn9, mlp9, mlp10} carrying
# .718 of the mass. The theseus registry still carries a FLOOR_CAVEAT on it.
# This runs the matched-rank control at that exact cell.
#
# question @ mlp11, rank-2 |lambda| (S1597's exact config) vs a random
# orthonormal rank-2 basis, 3 disjoint 96-row samples each, no refitting.
# Share = top-4 signed positive mass / total positive mass, as S1597 computed it.
#
# Registered predictions:
#   pred_a a floor exists here too: the random top-4 shares >= 2 of 4 components
#          with the |lambda| top-4.
#   pred_b the CIRCUIT components are rule-specific, not floor: attn10 and attn9
#          are both in the |lambda| top-4 and NEITHER is in the random top-4, in
#          >= 2 of 3 samples.
#   pred_c "72% in four components" is NOT by itself evidence of sparsity: the
#          RANDOM arm's own top-4 share is >= .40. If random also concentrates
#          most of its mass in 4 components, the .718 figure needs the floor-
#          corrected version quoted beside it wherever it appears.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'writer_floor_question_results.json'
NR = 960
SITE = 17
RANK = 8
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
EDIT = {'set': set(), 'V': None, 'mu': None}   # mu: {name: [2]}
FIN = {'on': False, 'V': None, 'mu': None}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def mk_cproj_hook(L):
    def hook(mod, args, output):
        nm = f'attn{L}'
        if nm not in EDIT['set']:
            return None
        o = output.float()
        pv = o @ EDIT['V']                       # [B,T,2]
        o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return o.to(output.dtype)
    return hook


def mk_mlp_hook(L):
    def hook(mod, args, output):
        o = None
        nm = f'mlp{L}'
        if nm in EDIT['set']:
            o = output.float()
            pv = o @ EDIT['V']
            o = o - (pv - EDIT['mu'][nm]) @ EDIT['V'].T
        return None if o is None else o.to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    if FIN['on']:
        xf = x.float()
        pv = xf @ FIN['V']
        x = (xf - (pv - FIN['mu']) @ FIN['V'].T).to(x.dtype)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def capture_fwd(idx, V2, lam2, acc, pm):
    """Exact manual forward through layer SITE, accumulating projections of
    every component output onto V2 (global + class sums), head-grain scores,
    mean_s, and the reconstruction check. pm: [B,T] class mask."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    vmask = torch.ones(B, T, dtype=torch.bool, device=DEV)
    vmask[:, :64] = False
    vf = vmask.reshape(-1); pf = pm.reshape(-1)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))

    def add(nm, o):
        pv = (o.float().reshape(-1, D) @ V2)      # [N,2]
        acc['sum'][nm] += pv[vf].sum(0)
        acc['csum'][nm] += pv[pf].sum(0)

    add('x0', x0)
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        qp = at.c_q(xin).view(B, T, 9, 128).float()
        kp = at.c_k(xin).view(B, T, 9, 128).float()
        q2p = at.c_q2(xin).view(B, T, 9, 128).float()
        k2p = at.c_k2(xin).view(B, T, 9, 128).float()
        cos, sin = at.rotary(qp)
        q = are(F.rms_norm(qp, (128,)), cos, sin)
        k = are(F.rms_norm(kp, (128,)), cos, sin)
        q2 = are(F.rms_norm(q2p, (128,)), cos, sin)
        k2 = are(F.rms_norm(k2p, (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        ao = at.c_proj(y.reshape(B, T, D))
        add(f'attn{L}', ao)
        # head grain: y_h @ Wp_h.T projected on V2
        Wp = at.c_proj.weight.float()             # [D, D]
        for hh in range(9):
            M = Wp[:, hh * 128:(hh + 1) * 128].T @ V2      # [128,2]
            pv = (y[:, :, hh].float().reshape(-1, 128) @ M)
            acc['hsum'][L][hh] += pv[vf].sum(0)
            acc['hcsum'][L][hh] += pv[pf].sum(0)
        x = xm + ao
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
    P = x
    acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
    acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))


def _stack_order():
    order = ['x0']
    for L in range(18):
        order += [f'attn{L}', f'mlp{L}']
    return order


def _layer_of(name):
    if name == 'x0':
        return -1
    return int(name[4:]) if name.startswith('attn') else int(name[3:])


@torch.no_grad()
def depth_curve(rows, V2, lam2, mask_v):
    """Per-component signed class contribution, summed in stack order."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    rk = V2.shape[1]          # infer from the basis; cells differ in rank (2 vs 8)
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]
        pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)

    # exact depth-decay coefficients from the block lambdas (same as S1601)
    lam0 = [float(blk.lambdas[0]) for blk in H]
    lam1 = [float(blk.lambdas[1]) for blk in H]
    coef = {}
    for l in range(18):
        c = 1.0
        for k in range(l + 1, 18):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(18):
        tx0 = lam0[k] * tx0 + lam1[k]
    coef['x0'] = tx0

    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    signed = {c: (coef[c] * (cmu[c] - mu[c])).sum().item() for c in comps}

    order = _stack_order()
    steps = [signed[c] for c in order]
    cum, run = [], 0.0
    for s in steps:
        run += s
        cum.append(run)
    peak_i = max(range(len(cum)), key=lambda i: cum[i])
    drop_i = min(range(len(steps)), key=lambda i: steps[i])
    peak, final = cum[peak_i], cum[-1]
    drawdown = (peak - final) / peak if peak > 0 else float('nan')
    # head grain, computed exactly as S1597/S1598: signed class-minus-global head
    # contribution, |.|-summed over the slice directions, times the depth coefficient
    hg = {}
    for L in range(18):
        hd = {h: float((coef[f'attn{L}'] * (acc['hcsum'][L][h] / max(acc['cn'], 1)
                        - acc['hsum'][L][h] / max(acc['n'], 1))).abs().sum())
              for h in range(9)}
        rk = sorted(hd.items(), key=lambda kv: -kv[1])
        hg[L] = {'top_head': f"{L}.{rk[0][0]}", 'top_val': rk[0][1],
                 'second_val': rk[1][1],
                 'ratio': rk[0][1] / max(rk[1][1], 1e-12)}

    return {'head_grain': hg,
            'order': order, 'steps': steps, 'cum': cum,
            'peak_component': order[peak_i], 'peak_layer': _layer_of(order[peak_i]),
            'peak_value': peak, 'final_value': final, 'drawdown': drawdown,
            'largest_drop_component': order[drop_i],
            'largest_drop_value': steps[drop_i],
            'recon_rel_err': rec_err, 'class_n': acc['cn'], 'signed': signed}



# LOCAL CORPUS -- NO NETWORK.  Decision recorded 2026-08-27 06:08 against a
# criterion registered BEFORE the fact ("if still in rows_cache past 900 s, a
# single FineWeb stream is unaffordable").  Three configurations died in
# rows_cache without reaching compute: skips [15000,20000,25000] (1740 s),
# [80,300,600] (1526 s), and a SINGLE 288-row stream (1026 s, 0 retries -- not
# erroring, just too slow).  The HF cache holds ZERO fineweb parquet files, so
# streaming re-downloads every time, and there is no HF_TOKEN on this box.
#
# Rows now come from bilin18_eval_tokens_large.pt (512, 513) int64, already on
# disk and loaded at import by bilin18_joint_removal as FW.
#
# LIMITATION, stated because it is real: FW is the DEDUP set that
# census_lib.fineweb_rows EXCLUDES, so these rows are not "fresh" in this
# program's sense, and a share computed on them is NOT directly comparable to
# S1597's .718 figure.  What survives intact is every REGISTERED bar: pred_a,
# pred_b and pred_c are all WITHIN-RUN comparisons between the |lambda| arm and
# the random arm on IDENTICAL rows, so the floor question is answered exactly as
# designed.  Only the incidental cross-reference to .718 weakens, and the writeup
# must say so rather than quoting the two side by side.
CHUNKS = 3
ROWS_PER_CHUNK = 96
LOCAL_ROWS = 'bilin18_eval_tokens_large.pt'
SITE_Q = 11
RANK_Q = 2
QPAT = r'^\?$| \?$'
CIRCUIT = {'attn10', 'attn9'}          # S1597's certified attention writers
TOP = 4

@torch.no_grad()
def main():
    import os, statistics
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    mask_v = rx(QPAT)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE_Q].mlp.Left.weight.float(); Rw = H[SITE_Q].mlp.Right.weight.float()
    Dw = H[SITE_Q].mlp.Down.weight.float()
    Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK_Q]          # S1597's rule
    lam_V, lam_l = V[:, o].contiguous(), lam[o].contiguous()
    gen = torch.Generator(device=DEV).manual_seed(1729)
    rnd_V, _ = torch.linalg.qr(torch.randn(D, RANK_Q, device=DEV, generator=gen))
    print(f"question@mlp{SITE_Q} rank-{RANK_Q} |lambda| eigs "
          f"{[round(float(x), 3) for x in lam_l]}  (S1597 ref: +144.9/-73.8, top4 share .718)",
          flush=True)

    _all = torch.load(PT + LOCAL_ROWS, map_location='cpu')[
        :CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    assert _all.shape[0] == CHUNKS * ROWS_PER_CHUNK, f'short load {_all.shape}'
    rows_cache = {c: _all[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)}
    print(f'LOCAL {LOCAL_ROWS} -> {CHUNKS} disjoint chunks of {ROWS_PER_CHUNK} rows '
          f'(no network; rows are the FW dedup set, NOT fresh -- see docstring)', flush=True)
    arms = {'lambda': (lam_V, lam_l), 'random': (rnd_V.contiguous(), torch.ones(RANK_Q, device=DEV))}
    res = {}
    for name, (V2, lam2) in arms.items():
        per = []
        for skip in range(CHUNKS):
            r = depth_curve(rows_cache[skip], V2, lam2, mask_v)
            signed = r.pop('signed')
            pos = sorted([c for c in signed if signed[c] > 0], key=lambda c: -signed[c])
            tot_pos = sum(signed[c] for c in pos)
            top = pos[:TOP]
            share = sum(signed[c] for c in top) / max(tot_pos, 1e-9)
            per.append({'top4': top, 'share': round(share, 4), 'class_n': r['class_n'],
                        'recon': r['recon_rel_err']})
            print(f"  {name:7s} skip={skip:6d} n={r['class_n']:3d} top4={top} "
                  f"share={share:.4f} recon={r['recon_rel_err']:.1e}", flush=True)
        res[name] = per

    # consensus top-4 per arm (rank-weighted across samples)
    def consensus(per):
        cnt = {}
        for p in per:
            for i, c in enumerate(p['top4']):
                cnt[c] = cnt.get(c, 0) + (TOP - i)
        return sorted(cnt, key=lambda c: -cnt[c])[:TOP]
    C = {n: consensus(res[n]) for n in arms}
    floor = set(C['random'])
    overlap = len(set(C['lambda']) & floor)
    lam_share = statistics.mean([p['share'] for p in res['lambda']])
    rnd_share = statistics.mean([p['share'] for p in res['random']])
    circuit_clean = sum(1 for p in res['lambda']
                        if CIRCUIT <= set(p['top4'])
                        and not (CIRCUIT & set(res['random'][res['lambda'].index(p)]['top4'])))
    corrected = [c for c in C['lambda'] if c not in floor]

    pa = overlap >= 2
    pb = circuit_clean >= 2
    pc = rnd_share >= 0.40

    out = {'config': {'class': 'question', 'site': SITE_Q, 'rank': RANK_Q,
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK, 'row_source': LOCAL_ROWS, 'rows_are_fresh': False, 'top': TOP,
                      'S1597_reference': {'top4': ['attn10', 'attn9', 'mlp9', 'mlp10'],
                                          'share': 0.718}},
           'arms': res,
           'consensus': C, 'floor': sorted(floor),
           'lambda_top4_floor_corrected': corrected,
           'overlap_lambda_random_of4': overlap,
           'mean_share': {'lambda': round(lam_share, 4), 'random': round(rnd_share, 4)},
           'circuit_clean_samples': circuit_clean,
           'predictions': {'pred_a_floor_exists_overlap_ge2': bool(pa),
                           'pred_b_circuit_rule_specific_2of3': bool(pb),
                           'pred_c_random_share_ge40pct': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\nconsensus lambda {C['lambda']}\nconsensus random {C['random']}", flush=True)
    print(f"overlap {overlap}/4 | floor-corrected lambda top4 -> {corrected}", flush=True)
    print(f"mean share: lambda {lam_share:.4f} vs random {rnd_share:.4f} "
          f"(S1597 reported .718)", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
