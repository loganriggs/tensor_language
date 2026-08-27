# share_sample_size_sweep: IS THE .718 -> .545 GAP A SAMPLE-SIZE ARTIFACT OF MY
# DESIGN, OR A REAL DISCREPANCY IN S1597?
#
# S1620 measured the question@mlp11 share at .545 on canonical FineWeb rows
# against S1597's published .718, and pronouns@mlp17 at .581 against S1598's .482.
# Neither reproduced. But S1597 used NR=960 rows while S1620 used 3 chunks of 160
# (n = 56-80 question positions per chunk). Share is a ratio over 37 components:
# with few class positions the class mean cmu is noisily estimated, delta is
# noise-spread, and the share should be DEPRESSED. S1620 recorded an explicit
# guard that the gap must not be cited as a discrepancy until this is run.
#
# Single sample at each of three sizes -- NO chunking, so each is one estimate of
# the same quantity at increasing n -- on the same canonical rows
# (.rowcache/fineweb_n480_skip80.pt, authority pinned_local_ordered_manifest;
# .rowcache_shadow untouched). Absolute attribution mass, slice_writers.py:216.
# Both cells, each with its matched-rank random arm at the same size.
#
# Registered predictions:
#   pred_a MONOTONE IN n: the question@mlp11 |lambda| share rises across
#          160 -> 320 -> 480 rows (non-decreasing at every step).
#   pred_b MOVES TOWARD THE PUBLISHED VALUE: |share(480) - .718| is strictly
#          smaller than |share(160) - .718|.
#   pred_c THE NULL RISES TOO: the matched-rank random share at question@mlp11
#          also increases from 160 to 480 -- confirming this is an ESTIMATION
#          effect on the statistic, not signal appearing with more data.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'share_sample_size_sweep_results.json'
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
SIZES = [160, 320, 480]
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
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
    import hashlib
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL fineweb_n480_skip80.pt sizes {SIZES} (receipt {rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    out_cells = {}

    for cname, cfg in CELLS.items():
        site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
        mask_v = rx(cfg['pat'])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
        Dw = H[site].mlp.Down.weight.float()
        Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        o = lam.abs().argsort(descending=True)[:rank]
        gen = torch.Generator(device=DEV).manual_seed(1729)
        rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
        arms = {'lambda': (V[:, o].contiguous(), lam[o].contiguous()),
                'random': (rnd.contiguous(), torch.ones(rank, device=DEV))}
        print(f"\n=== {cname} rank-{rank} TOP={TOP} (published {cfg['published']}) ===", flush=True)
        cell = {'published': cfg['published'], 'by_size': {}}
        for nrows in SIZES:
            rows = raw[:nrows, :T + 1].contiguous()
            rec = {}
            for aname, (V2, lam2) in arms.items():
                delta, cn, recon = abs_mass(rows, V2, lam2, mask_v)
                ranked = sorted(delta, key=lambda c: -delta[c])
                tot = sum(delta.values())
                rec[aname] = {'share': round(sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9), 4),
                              'class_n': cn, 'recon': recon, 'top': ranked[:TOP]}
            cell['by_size'][str(nrows)] = rec
            print(f"  n_rows={nrows:4d} class_n={rec['lambda']['class_n']:5d}  "
                  f"lambda={rec['lambda']['share']:.4f}  null={rec['random']['share']:.4f}", flush=True)
        out_cells[cname] = cell

    q = out_cells['question@mlp11']['by_size']
    ls = [q[str(n)]['lambda']['share'] for n in SIZES]
    ns = [q[str(n)]['random']['share'] for n in SIZES]
    pa = all(ls[i] <= ls[i + 1] for i in range(len(ls) - 1))
    pb = abs(ls[-1] - 0.718) < abs(ls[0] - 0.718)
    pc = ns[-1] > ns[0]

    out = {'config': {'sizes': SIZES, 'cells': {k: {kk: vv for kk, vv in v.items() if kk != 'pat'}
                                                for k, v in CELLS.items()},
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, pinned_local_ordered_manifest)',
                      'rows_are_fresh': True, 'receipt': 'fineweb_oracle_v2_receipt.json',
                      'statistic': 'absolute attribution mass (slice_writers.py:216)'},
           'cells': out_cells,
           'question_lambda_by_size': ls, 'question_null_by_size': ns,
           'S1620_reference': {'question_lambda': 0.545, 'question_null': 0.426,
                               'published': 0.718},
           'predictions': {'pred_a_lambda_monotone_in_n': bool(pa),
                           'pred_b_moves_toward_published': bool(pb),
                           'pred_c_null_rises_too': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  question lambda by size {ls}  |  null by size {ns}", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
