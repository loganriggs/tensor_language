# headgrain_extension: DOES THE HEAD-GRAIN LAW HOLD BEYOND ITS TWO ORIGINAL CELLS?
#
# S1597/S1598 stated it: "every attention writer of a certified eigen slice
# resolves at head grain to a certified circuit head of that class". S1608
# controlled it against a matched-rank random basis and it PASSED decisively --
# 3.1x, 6.9x and 18.8x separation at circuit layers, published ratios reproducing
# (6.8:1 -> 6.93, 20:1 -> 33.15), with a clean negative control at question@mlp9
# where the law correctly says nothing.
#
# But S1608 rests on TWO cells. LESSONS 17 was just written after a
# PRE-REGISTERED correlation collapsed from rho .673 to .018 on disjoint units,
# so a two-cell law deserves more units before it is leaned on. NOTE the
# difference and why this is not the same situation: S1608 measured LARGE effects
# (3-19x) with a mechanistic prediction and a negative control, not a rank
# correlation selected on 10 points. This run tests generality, not significance.
#
# SIX classes with certified circuit heads, none of them the two original cells:
#   colon (12.6), months (14.7), close_paren (13.8),
#   digits (7.3, 6.5, 12.6, 11.5), comma (6.5, 15.2, 9.5, 11.7),
#   and (10.5, 16.8, 7.3, 9.5)
# Slice site is NOT established for these classes, so BOTH mlp11 and mlp17 are
# tested and reported separately rather than committing to one.
# |lambda|-r8 slice (the published rule) vs a matched-rank random arm on identical
# rows. A "cell" is a (class, site, certified-layer) triple: at each layer that
# contains a certified head for that class, does the arm's top head MATCH it?
# Local curated_rows.pt 3 x 333, seed 1729, absolute-mass head grain exactly as
# slice_writers.py:224 computes it.
#
# Registered predictions:
#   pred_a the |lambda| arm's top head MATCHES the certified head in >= 50% of
#          (class, site, layer) cells.
#   pred_b the RANDOM arm matches in <= 20% of the same cells.
#   pred_c the |lambda| match rate is at least 2x the random match rate.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'headgrain_extension_results.json'
NR = 960
RANK = 8
SEED = 1729
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


CERTIFIED = {
    'colon':       ['12.6'],
    'months':      ['14.7'],
    'close_paren': ['13.8'],
    'digits':      ['7.3', '6.5', '12.6', '11.5'],
    'comma':       ['6.5', '15.2', '9.5', '11.7'],
    'and':         ['10.5', '16.8', '7.3', '9.5'],
}
PATTERNS = {
    'colon':       r'^:$|^ :$',
    'months':      r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$',
    'close_paren': r'^\)$|^ \)$',
    'digits':      r'^ ?[0-9]+$',
    'comma':       r'^,$|^ ,$',
    'and':         r'^ and$',
}
SITES = [11, 17]
CHUNKS, ROWS_PER_CHUNK = 3, 333


@torch.no_grad()
def head_grain(rows, V2, mask_v):
    """Per-layer (top head, ratio) using the S1597/S1598 head-grain statistic:
    |coef * (class-mean minus global-mean)| summed over slice directions."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    rk = V2.shape[1]
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    lam2 = torch.ones(rk, device=DEV)
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        capture_fwd(bb[:, :-1].to(DEV).contiguous(), V2, lam2, acc, pm)
    lam0 = [float(b.lambdas[0]) for b in H]
    coef = {}
    for l in range(18):
        c = 1.0
        for kk in range(l + 1, 18):
            c *= lam0[kk]
        coef[f'attn{l}'] = c
    out = {}
    for L in range(18):
        hd = {h: float((coef[f'attn{L}'] * (acc['hcsum'][L][h] / max(acc['cn'], 1)
                        - acc['hsum'][L][h] / max(acc['n'], 1))).abs().sum())
              for h in range(9)}
        rk_ = sorted(hd.items(), key=lambda kv: -kv[1])
        out[L] = {'top': f"{L}.{rk_[0][0]}",
                  'ratio': rk_[0][1] / max(rk_[1][1], 1e-12)}
    return out, acc['cn']


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    g = torch.Generator(device=DEV).manual_seed(SEED)
    rnd, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=g))
    rnd = rnd.contiguous()

    cells = []          # one entry per (class, site, certified-layer)
    detail = {}
    for cname, heads in CERTIFIED.items():
        mask_v = rx(PATTERNS[cname])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        for site in SITES:
            Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
            Dw = H[site].mlp.Down.weight.float()
            Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
            S = 0.5 * (Q + Q.T)
            lam, V = torch.linalg.eigh(S)
            o = lam.abs().argsort(descending=True)[:RANK]
            arms = {'lambda': V[:, o].contiguous(), 'random': rnd}
            got = {}
            for aname, V2 in arms.items():
                per = []
                for ch in chunks:
                    hg, cn = head_grain(ch, V2, mask_v)
                    per.append(hg)
                got[aname] = per
            key = f'{cname}@mlp{site}'
            detail[key] = {}
            for hd in heads:
                L = int(hd.split('.')[0])
                lam_tops = [got['lambda'][i][L]['top'] for i in range(CHUNKS)]
                rnd_tops = [got['random'][i][L]['top'] for i in range(CHUNKS)]
                lam_hit = sum(1 for t in lam_tops if t == hd) >= 2      # majority of 3
                rnd_hit = sum(1 for t in rnd_tops if t == hd) >= 2
                lam_ratio = sum(got['lambda'][i][L]['ratio'] for i in range(CHUNKS)) / CHUNKS
                rnd_ratio = sum(got['random'][i][L]['ratio'] for i in range(CHUNKS)) / CHUNKS
                cells.append({'cell': f'{key} L{L}', 'certified': hd,
                              'lambda_tops': lam_tops, 'random_tops': rnd_tops,
                              'lambda_hit': lam_hit, 'random_hit': rnd_hit,
                              'lambda_ratio': round(lam_ratio, 3),
                              'random_ratio': round(rnd_ratio, 3)})
                detail[key][hd] = cells[-1]
                print(f"  {key:20s} L{L:<2d} cert={hd:6s} lambda={lam_tops} "
                      f"{'HIT' if lam_hit else '   '}  random={rnd_tops} "
                      f"{'hit' if rnd_hit else '   '}", flush=True)

    nL = sum(1 for c in cells if c['lambda_hit'])
    nR = sum(1 for c in cells if c['random_hit'])
    N = len(cells)
    rateL, rateR = nL / N, nR / N
    pa = rateL >= 0.50
    pb = rateR <= 0.20
    pc = rateL >= 2 * rateR if rateR > 0 else rateL > 0

    out = {'config': {'classes': list(CERTIFIED), 'sites': SITES, 'rank': RANK,
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK, 'seed': SEED,
                      'row_source': 'curated_rows.pt', 'rows_are_fresh': False,
                      'hit_rule': 'top head == certified head in a majority of 3 chunks'},
           'cells': cells, 'n_cells': N,
           'lambda_hits': nL, 'random_hits': nR,
           'lambda_rate': round(rateL, 4), 'random_rate': round(rateR, 4),
           'S1608_reference': 'two cells: question@mlp11 18.8x, pronouns@mlp17 3.1x/6.9x',
           'predictions': {'pred_a_lambda_ge50pct': bool(pa),
                           'pred_b_random_le20pct': bool(pb),
                           'pred_c_lambda_2x_random': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  cells={N}  lambda hits {nL} ({rateL:.1%})  random hits {nR} ({rateR:.1%})", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
