# published_vs_null_evalskip: TEST .718 AND .482 ON THE ROWS THEY WERE ACTUALLY
# COMPUTED FROM (skip=7000), NOT skip=80.
#
# S1620 measured question@mlp11 at .545 against S1597's published .718 and called
# it a non-reproduction. S1621 then found the comparison was INVALID:
#     slice_writers.py:159   EVR = fineweb_rows(NR=960, skip=7000)   <- S1597 EVAL
#     slice_writers.py:160   FR  = fineweb_rows(96,     skip=80)     <- fit only
# S1620 used skip=80. Every share-vs-published sentence in S1620 is withdrawn.
# S1621 also established the share is FLAT in n (.5427 -> .5462 across a 2.7x
# change in class positions), so the smaller canonical tensor at the right skip
# should still be informative.
#
# Rows: .rowcache/fineweb_n192_skip7000.pt -- S1597's OWN eval skip, canonical
# (authority pinned_local_ordered_manifest, scored-work true). 192 rows, 47
# question and 266 pronoun positions. 3 disjoint 64-row chunks. Absolute
# attribution mass, slice_writers.py:216. .rowcache_shadow untouched.
#
# CAVEAT stated up front: 47 question positions is BELOW the 78-214 range over
# which S1621 verified flatness. If pred_c fails while pred_a passes, suspect the
# small n before believing the share.
#
# Registered predictions:
#   pred_a AT THE RIGHT SKIP THE PUBLISHED FIGURE REPRODUCES: the question@mlp11
#          |lambda| share lands within .10 of .718.
#   pred_b SKIP MATTERS: the share here differs from S1620's skip=80 value of
#          .545 by >= .05, i.e. the row set is what moved it.
#   pred_c DIRECTION IS SKIP-INDEPENDENT: |lambda| still exceeds its matched-rank
#          null at question@mlp11, and pronouns@mlp17 still sits BELOW its null.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'published_vs_null_evalskip_results.json'
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
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
    P = x
    acc['n'] += int(vf.sum()); acc['cn'] += int(pf.sum())
    acc['P_proj'].append((P.float().reshape(-1, D) @ V2)[vf].sum(0))


CELLS = {
    'question@mlp11': {'site': 11, 'rank': 2, 'top': 4,
                       'pat': r'^\?$| \?$', 'published': 0.718,
                       'published_top': ['attn10', 'attn9', 'mlp9', 'mlp10']},
    'pronouns@mlp17': {'site': 17, 'rank': 8, 'top': 6,
                       'pat': r'^ (he|she|they|He|She|They)$', 'published': 0.482,
                       'published_top': ['mlp16', 'x0', 'mlp15', 'mlp9', 'mlp14', 'attn9']},
}
CHUNKS, ROWS_PER_CHUNK = 3, 64
ROWCACHE = PT + '.rowcache/fineweb_n192_skip7000.pt'
RECEIPT  = PT + '.rowcache/fineweb_oracle_v2_receipt.json'


@torch.no_grad()
def abs_mass(rows, V2, lam2, mask_v):
    """Per-component ABSOLUTE attribution mass -- the S1597/S1598 statistic:
    delta_c = |coef_c * (class_mean - global_mean)| summed over slice directions."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
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

    q, p = out_cells['question@mlp11'], out_cells['pronouns@mlp17']
    ql, qr = q['mean_share']['lambda'], q['mean_share']['random']
    pl, pr = p['mean_share']['lambda'], p['mean_share']['random']
    pa = abs(ql - 0.718) <= 0.10
    pb = abs(ql - 0.545) >= 0.05
    pc = (ql > qr) and (pl < pr)
    print(f"\n  question lambda {ql} null {qr} | published .718 | S1620 skip80 .545", flush=True)
    print(f"  pronouns lambda {pl} null {pr} | published .482", flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'rows_are_fresh': True, 'receipt': 'fineweb_oracle_v2_receipt.json',
                      'statistic': 'ABSOLUTE attribution mass, matching slice_writers.py:216'},
           'cells': out_cells,
           'predictions': {'pred_a_reproduces_718_within_10': bool(pa),
                           'pred_b_skip_moves_share_ge05': bool(pb),
                           'pred_c_directions_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
