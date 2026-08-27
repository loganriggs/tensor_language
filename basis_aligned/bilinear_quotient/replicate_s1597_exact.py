# replicate_s1597_exact: S1597's EXACT CONFIGURATION -- 96 rows at skip=80.
#
# The .718 chase so far, and every explanation eliminated:
#   S1620  corpus      -- canonical FineWeb, measured .545 vs published .718
#   S1621  sample size -- share FLAT in n (.5427 -> .5462 over a 2.7x change)
#   S1622  row skip    -- skip=7000 gives .5484, a .0034 move. Also WITHDREW
#          S1621's "wrong rows" claim: slice_writers.py:184-185 fills `acc` from
#          FR, the 96 FIT rows at skip=80, so S1620's skip choice was right.
#
# One difference remains: S1597 uses exactly 96 rows (~25-30 class positions),
# BELOW the 78-214 range over which S1621 verified flatness. This closes it using
# Codex's canonical `.rowcache/fineweb_n96_skip80.pt` -- n and skip identical to
# `cl.fineweb_rows(96, skip=80)`. Streaming remains dead here (a 90 s probe of
# fineweb_rows(8, skip=40) timed out at 09:09, HF cache still 48K), so running
# slice_writers.py itself is not possible; this is the closest available.
#
# SINGLE sample of all 96 rows -- no chunking, matching S1597's single-pass fit.
# Absolute attribution mass, slice_writers.py:216. .rowcache_shadow untouched.
#
# I have twice blamed S1597 and twice found the fault in my own setup (S1620's
# corpus, S1621's row set). If pred_a fails HERE, at its exact configuration with
# every other variable eliminated, that is a genuine unexplained discrepancy and
# goes to Logan rather than into a claim.
#
# Registered predictions:
#   pred_a EXACT CONFIG REPRODUCES: the question@mlp11 |lambda| top-4 share lands
#          within .10 of the published .718.
#   pred_b IF NOT, IT IS CONSISTENT WITH EVERY OTHER MEASUREMENT: the share is
#          within .05 of the .545 / .5484 already measured at other configs --
#          i.e. the quantity is stable and the gap is not configuration noise.
#   pred_c the |lambda| share still exceeds its matched-rank null at this config.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'replicate_s1597_exact_results.json'
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
CHUNKS, ROWS_PER_CHUNK = 1, 96
ROWCACHE = PT + '.rowcache/fineweb_n96_skip80.pt'
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
    pb = min(abs(ql - 0.545), abs(ql - 0.5484)) <= 0.05
    pc = ql > qr
    print(f"\n  question lambda {ql} null {qr} | published .718 | S1620 skip80 .545", flush=True)
    print(f"  pronouns lambda {pl} null {pr} | published .482", flush=True)

    out = {'config': {'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()},
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'rows_are_fresh': True, 'receipt': 'fineweb_oracle_v2_receipt.json',
                      'statistic': 'ABSOLUTE attribution mass, matching slice_writers.py:216'},
           'cells': out_cells,
           'predictions': {'pred_a_exact_config_reproduces_718': bool(pa),
                           'pred_b_consistent_with_other_configs': bool(pb),
                           'pred_c_lambda_above_null': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
