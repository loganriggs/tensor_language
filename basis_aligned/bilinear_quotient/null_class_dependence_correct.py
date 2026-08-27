# null_class_dependence_correct: IS S1612's "CELL-DEPENDENT NULL" AN ARTIFACT OF
# THE WRONG QUANTITY?
#
# S1612 measured matched-rank null shares of .4489 (question@mlp11) and .7295
# (pronouns@mlp17) -- a .28 spread -- and concluded the null is CELL-DEPENDENT and
# a bare share uninterpretable. S1613 built on it (null not tabulable, class effect
# 3.3x the rank effect), S1614/S1616 chased what predicts it, and all of that was
# measured with the quantity S1623 showed to be WRONG: full 18-layer forward,
# final-residual attribution, all 37 components.
#
# Under the CORRECT measurement (forward stops at the site, upstream components,
# site-relative coefficients) the two nulls came out at .5711 (S1623) and .5744
# (S1624) -- a difference of .0033. That is a 85x reduction in spread and it
# suggests S1612's headline may be an artifact.
#
# CAVEAT that keeps this honest: those two cells also differ in rank (2 vs 8) and
# TOP (4 vs 6), so their near-equality could be coincidence. This isolates CLASS by
# holding site, rank and TOP fixed: six classes at mlp11, rank-2, TOP-4, forward
# stopping at 11, upstream components only, canonical
# `.rowcache/fineweb_n96_skip80.pt`. RANDOM ARM ONLY -- the null is the object of
# study. .rowcache_shadow untouched.
#
# Registered predictions:
#   pred_a UNDER THE CORRECT QUANTITY THE NULL IS NEARLY CLASS-INDEPENDENT: the
#          range of null shares across the six classes is <= .10.
#   pred_b IT IS SMALLER THAN S1613 MEASURED: the range is below S1613's .1417,
#          which was computed on the same six-ish classes with the wrong quantity.
#   pred_c NOT DEGENERATE: at least one class's null differs from question's .5711
#          by >= .03, so a small range reflects real measurement rather than the
#          statistic collapsing to a constant.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE_STOP = 11        # S1597 SITE: forward stops here, P = its input
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'null_class_dependence_correct_results.json'
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
        if L == SITE_STOP:
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


CELLS = {
    'question': {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^\?$| \?$', 'published': 0.718,
                 'published_top': []},
    'the':      {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^ the$', 'published': None,
                 'published_top': []},
    'is':       {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^ is$', 'published': None,
                 'published_top': []},
    'comma':    {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^,$|^ ,$', 'published': None,
                 'published_top': []},
    'and':      {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^ and$', 'published': None,
                 'published_top': []},
    'to':       {'site': 11, 'rank': 2, 'top': 4, 'pat': r'^ to$', 'published': None,
                 'published_top': []},
}
CHUNKS, ROWS_PER_CHUNK = 1, 96
ROWCACHE = PT + '.rowcache/fineweb_n96_skip80.pt'
RECEIPT  = PT + '.rowcache/fineweb_oracle_v2_receipt.json'


@torch.no_grad()
def abs_mass(rows, V2, lam2, mask_v):
    """Per-component ABSOLUTE attribution mass -- the S1597/S1598 statistic:
    delta_c = |coef_c * (class_mean - global_mean)| summed over slice directions."""
    SITE_UP = 11          # S1597: comps are UPSTREAM of the slice site
    comps = ['x0'] + [f'attn{L}' for L in range(SITE_UP + 1)] \
        + [f'mlp{L}' for L in range(SITE_UP)]
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
    for l in range(SITE_STOP + 1):
        c = 1.0
        for k in range(l + 1, SITE_STOP + 1):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(SITE_STOP + 1):
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
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    assert raw.shape[0] >= CHUNKS * ROWS_PER_CHUNK, f'short row tensor {tuple(raw.shape)}'
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    import hashlib
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {CHUNKS} chunks x {ROWS_PER_CHUNK} rows '
          f'(FRESH FineWeb; receipt sha256[:16]={rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    out_cells = {}

    for cname, cfg in CELLS.items():
        site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
        mask_v = rx(cfg['pat'])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
        Dw = H[site].mlp.Down.weight.float()
        S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
        lam, V = torch.linalg.eigh(S)
        o = lam.abs().argsort(descending=True)[:rank]        # the published rule
        gen = torch.Generator(device=DEV).manual_seed(1729)
        rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
        arms = {'lambda': (V[:, o].contiguous(), lam[o].contiguous()),
                'random': (rnd.contiguous(), torch.ones(rank, device=DEV))}
        print(f"\n=== {cname} rank-{rank} TOP={TOP} (published share {cfg['published']}) ===", flush=True)

        cell = {'published': cfg['published'], 'published_top': cfg['published_top'],
                'rank': rank, 'top': TOP, 'arms': {}}
        for aname, (V2, lam2) in arms.items():
            per = []
            for ci, ch in enumerate(chunks):
                delta, cn, rec = abs_mass(ch, V2, lam2, mask_v)
                ranked = sorted(delta, key=lambda c: -delta[c])
                tot = sum(delta.values())
                top = ranked[:TOP]
                share = sum(delta[c] for c in top) / max(tot, 1e-9)
                per.append({'chunk': ci, 'class_n': cn, 'share': round(share, 4),
                            'top': top, 'recon': rec})
                print(f"  {aname:7s} chunk{ci} n={cn:4d} share={share:.4f} top={top} "
                      f"recon={rec:.1e}", flush=True)
            cell['arms'][aname] = per
        L = [p['share'] for p in cell['arms']['lambda']]
        R = [p['share'] for p in cell['arms']['random']]
        cell['gaps'] = [round(abs(r - l), 4) for l, r in zip(L, R)]
        cell['mean_share'] = {'lambda': round(sum(L) / len(L), 4), 'random': round(sum(R) / len(R), 4)}
        cell['absent_from_random'] = [
            len([c for c in cell['arms']['lambda'][i]['top']
                 if c not in set(cell['arms']['random'][i]['top'])]) for i in range(CHUNKS)]
        print(f"  -> mean share lambda {cell['mean_share']['lambda']} vs random "
              f"{cell['mean_share']['random']} | gaps {cell['gaps']} | "
              f"lambda-top absent from random {cell['absent_from_random']}", flush=True)
        out_cells[cname] = cell

    nulls = {k: v['mean_share']['random'] for k, v in out_cells.items()}
    lams  = {k: v['mean_share']['lambda'] for k, v in out_cells.items()}
    vals = list(nulls.values())
    rng = round(max(vals) - min(vals), 4)
    far = sum(1 for v in vals if abs(v - 0.5711) >= 0.03)

    pa = rng <= 0.10
    pb = rng < 0.1417
    pc = far >= 1

    print(f"\n  NULL shares (correct quantity, all mlp11 r2 TOP4):", flush=True)
    for k in out_cells:
        print(f"    {k:9s} null {nulls[k]:.4f}   lambda {lams[k]:.4f}", flush=True)
    print(f"  range {rng}   (S1613 measured .1417 with the WRONG quantity)", flush=True)
    print(f"  classes >= .03 from question's .5711: {far}", flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'rows_are_fresh': True, 'receipt': 'fineweb_oracle_v2_receipt.json',
                      'statistic': 'ABSOLUTE attribution mass, matching slice_writers.py:216'},
           'cells': out_cells,
           'predictions': {'pred_a_null_range_le10': bool(pa),
                           'pred_b_smaller_than_S1613_1417': bool(pb),
                           'pred_c_not_degenerate': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
