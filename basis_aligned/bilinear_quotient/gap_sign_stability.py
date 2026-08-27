# gap_sign_stability: IS "THE CERTIFIED SLICE WRITES LESS CONCENTRATEDLY THAN A
# RANDOM BASIS" A REAL CLASS PROPERTY, OR NOISE?
#
# S1625 measured, at a FIXED cell (mlp11, rank-2, TOP-4) under the corrected
# quantity, the SIGNED gap = lambda_share - null_share for six classes:
#
#     to +.1717 | question +.1468 | and +.0639 | is +.0037 | comma -.0091 | the -.0360
#
# Two of them are NEGATIVE: the certified eigen slice concentrates its writers LESS
# than a random basis of the same rank. That was an UNREGISTERED observation on one
# chunk of 96 rows, so it is a hypothesis and nothing more (LESSONS 17). It matters
# because S1612 read exactly this sign at pronouns@mlp17 as "a positive structural
# claim about distributed writing"; if the sign is noise at mlp11, that reading
# needs re-examining.
#
# This tests SIGN STABILITY on THREE DISJOINT row chunks (160 rows each) from the
# canonical `.rowcache/fineweb_n480_skip80.pt`. Same site, rank and TOP for all six
# classes, so the only thing varying is the class and the rows. .rowcache_shadow
# untouched; no artifact of Codex's is read or written.
#
# Registered predictions (minimum effect sizes, not bare comparisons -- LESSONS 19):
#   pred_a SIGN IS STABLE WHERE THE EFFECT IS NOT NOISE: for the four classes whose
#          S1625 |gap| >= .03 (to, question, and, the) the signed gap holds the SAME
#          sign in all three disjoint chunks -- 12/12 agreements. The two near-zero
#          classes (comma -.0091, is +.0037) are registered IN ADVANCE as expected-
#          unstable and are excluded from this bar; their behaviour is reported but
#          not scored.
#   pred_b THE BELOW-NULL SIGN IS REAL: `the` has gap < 0 in all three chunks.
#   pred_c THE STRONG POSITIVE REPLICATES AT MAGNITUDE: `to` has gap >= +.10 in all
#          three chunks (S1625 saw +.1717).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE_STOP = 11        # S1597 SITE: forward stops here, P = its input
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gap_sign_stability_results.json'
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
CHUNKS, ROWS_PER_CHUNK = 3, 160        # three DISJOINT chunks
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
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
        cell['gaps'] = [round(l - r, 4) for l, r in zip(L, R)]   # SIGNED lambda - null
        cell['mean_share'] = {'lambda': round(sum(L) / len(L), 4), 'random': round(sum(R) / len(R), 4)}
        cell['absent_from_random'] = [
            len([c for c in cell['arms']['lambda'][i]['top']
                 if c not in set(cell['arms']['random'][i]['top'])]) for i in range(CHUNKS)]
        print(f"  -> mean share lambda {cell['mean_share']['lambda']} vs random "
              f"{cell['mean_share']['random']} | gaps {cell['gaps']} | "
              f"lambda-top absent from random {cell['absent_from_random']}", flush=True)
        out_cells[cname] = cell

    # S1625 single-chunk signed gaps (lambda - null), the hypothesis under test
    S1625_GAP = {'to': 0.1717, 'question': 0.1468, 'and': 0.0639,
                 'is': 0.0037, 'comma': -0.0091, 'the': -0.0360}
    SCORED = [c for c in CELLS if abs(S1625_GAP[c]) >= 0.03]      # to, question, and, the
    NOISE  = [c for c in CELLS if abs(S1625_GAP[c]) < 0.03]       # is, comma (reported, not scored)

    gaps = {k: out_cells[k]['gaps'] for k in out_cells}            # SIGNED, per chunk
    agree = {c: sum(1 for g in gaps[c] if (g > 0) == (S1625_GAP[c] > 0)) for c in SCORED}
    n_agree = sum(agree.values())

    pa = n_agree == 3 * len(SCORED)                                # 12/12 sign agreements
    pb = all(g < 0 for g in gaps['the'])                           # below-null sign is real
    pc = all(g >= 0.10 for g in gaps['to'])                        # strong positive replicates

    print(f"\n  SIGNED gap (lambda - null) per disjoint chunk, all mlp11 r2 TOP4:", flush=True)
    for k in out_cells:
        tag = 'SCORED' if k in SCORED else 'noise '
        gg = ' '.join(f'{g:+.4f}' for g in gaps[k])
        signs = ''.join('+' if g > 0 else '-' for g in gaps[k])
        print(f"    {k:9s} {tag}  S1625 {S1625_GAP[k]:+.4f} | chunks {gg} | signs {signs}"
              f"{'  <- FLIPPED' if k in SCORED and agree[k] != 3 else ''}", flush=True)
    print(f"  sign agreements on the four scored classes: {n_agree}/12", flush=True)
    print(f"  'the' below null in all 3: {pb}   |   'to' >= +.10 in all 3: {pc}", flush=True)
    print(f"  (near-zero classes {NOISE} were registered as expected-unstable and are "
          f"NOT scored)", flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'rows_are_fresh': True, 'receipt': 'fineweb_oracle_v2_receipt.json',
                      'statistic': 'ABSOLUTE attribution mass, matching slice_writers.py:216; gap is SIGNED lambda - null'},
           'cells': out_cells,
           'predictions': {'pred_a_sign_stable_12_of_12': bool(pa),
                           'pred_b_the_below_null_all_chunks': bool(pb),
                           'pred_c_to_gap_ge_10_all_chunks': bool(pc)},
           'signed_gaps_per_chunk': gaps,
           'S1625_reference_gaps': S1625_GAP,
           'scored_classes': SCORED, 'unscored_near_zero': NOISE,
           'sign_agreements': agree,
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
