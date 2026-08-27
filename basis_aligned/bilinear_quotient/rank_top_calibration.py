# rank_top_calibration: WHERE DOES THE SEPARATION STATISTIC STOP DISCRIMINATING?
#
# §1639 showed the separation count SATURATES at rank-8 TOP-6: six cells spanning
# 17-59 at rank-2 TOP-4 all land at 57-60, and ` at`@mlp11 goes from 17/60 (no signal)
# to 57/60 on nothing but a configuration change. That invalidated the depth and
# class-type results as general claims, and it leaves the bench without an answer to
# the practical question: which (rank, TOP) settings actually discriminate?
#
# This calibrates. ONE class (` at`) at ONE site (mlp11) -- the cell with the largest
# known configuration swing -- across a 3x2 grid of rank in {2,4,8} x TOP in {4,6}.
# 20 independent random bases x 3 disjoint 160-row chunks = 60 trials per cell,
# 6 cells. Holding class and site fixed means every difference is configuration.
#
# Known anchors: (rank 2, TOP 4) gave 17/60 with gap −.0075; (rank 8, TOP 6) gave
# 57/60 with gap +.0492.
#
# Registered predictions:
#   pred_a RANK DRIVES IT, NOT TOP: |sep(8,4) − sep(8,6)| < |sep(8,4) − sep(2,4)|,
#          i.e. changing rank at fixed TOP moves the statistic more than changing TOP
#          at fixed rank.
#   pred_b A USABLE NON-SATURATED REGIME EXISTS BEYOND rank-2 TOP-4: at least one of
#          the five non-(2,4) configurations gives separation <= 45/60, the ceiling
#          the function-word classes showed in §1636.
#   pred_c IT IS MONOTONE IN RANK AT FIXED TOP=4: sep(2,4) <= sep(4,4) <= sep(8,4).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE = 17             # SINGLE source of truth: forward stop, comps and coefs all read this
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rank_top_calibration_results.json'
NR = 960
# per-cell site/rank set in CELLS below
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
        if L == SITE:
            P = x                      # S1597: P is the SITE's INPUT residual
            acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
            acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))
            return                     # forward ENDS at the site (slice_writers.py:56)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
    P = x
    acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
    acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))


PAT_AT = r'^ at$'
GRID = [(2, 4), (2, 6), (4, 4), (4, 6), (8, 4), (8, 6)]
CELLS = {f'r{r}t{t}': {'site': 11, 'rank': r, 'top': t, 'certified': False, 'pat': PAT_AT}
         for r, t in GRID}
CLASS_PATS = CELLS          # the per-cell loop iterates this name
SITES = [11]
DEPTH_ORDER = list(CELLS)
CHUNKS, ROWS_PER_CHUNK = 3, 160        # three DISJOINT chunks
N_RANDOM = 20                          # seeds 1729+i; i=0 is the single seed S1625/S1627 used
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT  = PT + '.rowcache/fineweb_oracle_v2_receipt.json'


@torch.no_grad()
def abs_mass(rows, V2, lam2, mask_v):
    """Per-component ABSOLUTE attribution mass -- the S1597/S1598 statistic:
    delta_c = |coef_c * (class_mean - global_mean)| summed over slice directions."""
    comps = ['x0'] + [f'attn{L}' for L in range(SITE + 1)] \
        + [f'mlp{L}' for L in range(SITE)]
    rk = V2.shape[1]
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)

    lam0 = [float(b.lambdas[0]) for b in H]; lam1 = [float(b.lambdas[1]) for b in H]
    coef = {}
    for l in range(SITE + 1):
        c = 1.0
        for k in range(l + 1, SITE + 1):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(SITE + 1):
        tx0 = lam0[k] * tx0 + lam1[k]
    coef['x0'] = tx0

    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    delta = {c: (coef[c] * (cmu[c] - mu[c])).abs().sum().item() for c in comps}
    return delta, acc['cn'], rec_err


@torch.no_grad()
def main():
    import os, hashlib
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    assert raw.shape[0] >= CHUNKS * ROWS_PER_CHUNK, f'short row tensor {tuple(raw.shape)}'
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {CHUNKS} chunks x {ROWS_PER_CHUNK} '
          f'rows x {N_RANDOM} seeds (receipt sha256[:16]={rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    out_cells = {}

    global SITE
    for cname in DEPTH_ORDER:
        cfg = CELLS[cname]
        site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
        SITE = site                      # single global drives stop, comps and coefs
        n_comps = 1 + (site + 1) + site
        print(f'  --- {cname}: SITE={SITE}, {n_comps} upstream components '
              f'(TOP-{TOP} share is NOT comparable across sites; separation IS)',
              flush=True)
        mask_v = rx(cfg['pat'])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
        Dw = H[site].mlp.Down.weight.float()
        S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
        lam, V = torch.linalg.eigh(S)
        o = lam.abs().argsort(descending=True)[:rank]
        V2l, lam2l = V[:, o].contiguous(), lam[o].contiguous()

        lam_share = []
        for ci, ch in enumerate(chunks):
            delta, cn, rec = abs_mass(ch, V2l, lam2l, mask_v)
            ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
            lam_share.append(sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9))

        per_chunk_rnd = [[] for _ in range(CHUNKS)]
        all_trials = []            # every one of the 60 random shares, for the count
        seed0_rnd = []
        for i in range(N_RANDOM):
            gen = torch.Generator(device=DEV).manual_seed(1729 + i)
            rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
            for ci, ch in enumerate(chunks):
                delta, cn, rec = abs_mass(ch, rnd.contiguous(),
                                          torch.ones(rank, device=DEV), mask_v)
                ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
                sh = sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9)
                per_chunk_rnd[ci].append(sh)
                all_trials.append(sh)
                if i == 0:
                    seed0_rnd.append(sh)

        mean_rnd = [sum(v) / len(v) for v in per_chunk_rnd]
        null20 = sum(mean_rnd) / len(mean_rnd)
        null1 = sum(seed0_rnd) / len(seed0_rnd)
        gaps20 = [round(l - r, 4) for l, r in zip(lam_share, mean_rnd)]
        gaps1 = [round(l - r, 4) for l, r in zip(lam_share, seed0_rnd)]
        lo = min(lam_share); hi = max(lam_share)
        mean_gap = sum(l - r for l, r in zip(lam_share, mean_rnd)) / CHUNKS
        # directional separation: how many of the 60 trials the lambda arm beats,
        # measured in whichever direction the class actually points
        if mean_gap >= 0:
            sep = sum(1 for t in all_trials if lo > t)
        else:
            sep = sum(1 for t in all_trials if hi < t)
        out_cells[cname] = {'separation': sep, 'trials': len(all_trials),
                            'mean_gap': round(mean_gap, 4),
                            'certified': bool(cfg.get('certified', False)),
                            'random_range': [round(min(all_trials), 4), round(max(all_trials), 4)],
                            'lambda': [round(x, 4) for x in lam_share],
                            'null_20seed_per_chunk': [round(x, 4) for x in mean_rnd],
                            'null_20seed': round(null20, 4),
                            'null_seed1729': round(null1, 4),
                            'gaps_20seed': gaps20, 'gaps_seed1729': gaps1}
        print(f"  {cname:9s} {'CERTIFIED' if cfg.get('certified') else 'fresh    '} "
              f"sep {sep:2d}/{len(all_trials)} | mean gap {mean_gap:+.4f} | "
              f"lambda {[round(x,4) for x in lam_share]} | null20 {null20:.4f}", flush=True)

    NT = out_cells[DEPTH_ORDER[0]]['trials']
    sep = {k: out_cells[k]['separation'] for k in out_cells}
    gap = {k: out_cells[k]['mean_gap'] for k in out_cells}

    def K(r, t):
        return f'r{r}t{t}'

    d_rank = abs(sep[K(8, 4)] - sep[K(2, 4)])      # rank moves at fixed TOP=4
    d_top = abs(sep[K(8, 4)] - sep[K(8, 6)])       # TOP moves at fixed rank=8
    non24 = [k for k in sep if k != K(2, 4)]
    n_unsat = sum(1 for k in non24 if sep[k] <= 45)
    mono_rank = sep[K(2, 4)] <= sep[K(4, 4)] <= sep[K(8, 4)]

    pa = d_top < d_rank
    pb = n_unsat >= 1
    pc = mono_rank

    print(f"\n  ` at`@mlp11, rank x TOP grid, {N_RANDOM*CHUNKS} trials per cell:", flush=True)
    print(f"    {'':6s} {'TOP=4':>12s} {'TOP=6':>12s}", flush=True)
    for r in (2, 4, 8):
        row = '  '.join(f"{sep[K(r,t)]:2d}/60 {gap[K(r,t)]:+.3f}" for t in (4, 6))
        print(f"    rank{r:<2d}  {row}", flush=True)
    print(f"  moving RANK at fixed TOP=4:  |{sep[K(8,4)]} - {sep[K(2,4)]}| = {d_rank}", flush=True)
    print(f"  moving TOP  at fixed rank=8: |{sep[K(8,4)]} - {sep[K(8,6)]}| = {d_top}", flush=True)
    print(f"  non-(2,4) configs still discriminating (<=45/60): {n_unsat}/5", flush=True)
    print(f"  monotone in rank at TOP=4: {mono_rank}", flush=True)
    print(f"  (§1639 anchors: (2,4) gave 17/60 gap -.0075; (8,6) gave 57/60 gap +.0492)",
          flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'n_random': N_RANDOM, 'seeds': f'1729..{1729+N_RANDOM-1}',
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'statistic': 'ABSOLUTE attribution mass, corrected quantity; null is a 60-trial mean'},
           'cells': out_cells,
           'predictions': {'pred_a_rank_moves_more_than_top': bool(pa),
                           'pred_b_a_usable_regime_exists': bool(pb),
                           'pred_c_monotone_in_rank': bool(pc)},
           'grid': [list(g) for g in GRID], 'separation': sep, 'gap': gap,
           'delta_rank': d_rank, 'delta_top': d_top, 'n_unsaturated': n_unsat,
           'site': 11, 'class': 'at',

           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
