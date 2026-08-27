# head_grain_multiseed: §1597's HEADLINE NOW RESTS ENTIRELY ON HEAD-GRAIN — does it
# survive the control that killed the membership claim?
#
# §1628 refuted §1612's top-4 MEMBERSHIP control for attn10 (present in 53% of 60
# random trials, and in 2/3 chunks on §1612's own seed). §1628 explicitly did not
# test the head-grain claim, which is the stronger and now the LOAD-BEARING half of
# §1597's headline. As registered in theseus-bench/registry/circuits.json:
#
#     head_grain: attn10 = "10.5 (625, 20:1 over next)",  attn9 = "9.7 (221)"
#
# i.e. within layer 10, head 5 carries 20x the attribution mass of the next head.
# That number was computed with a single lambda slice and NO random control at all.
# The open question is whether a 20:1 within-layer concentration is what the slice
# does, or what ANY rank-2 basis does at this site.
#
# Design: question@mlp11 rank-2, corrected quantity (forward stops at 11, upstream
# comps, site-relative coefficients). Per-head absolute attribution mass within
# layer 10 (9 heads), for the lambda arm and for 20 INDEPENDENT random rank-2 bases
# (seeds 1729-1748) x 3 disjoint 160-row chunks = 60 control trials.
#
# This CANNOT retract anything by itself; §1597 is a published claim and any
# withdrawal goes to Logan, as with PENDING_RETRACTION_S1612.md.
#
# Registered predictions:
#   pred_a THE IDENTIFICATION HOLDS: head 5 is the top head of layer 10 in the
#          lambda arm on all three chunks.
#   pred_b THE CONCENTRATION IS REAL AND NOT A PROPERTY OF ANY BASIS: the lambda
#          arm's top:next head ratio exceeds the 95th percentile of the 60 random
#          trial ratios on all three chunks.
#   pred_c HEAD IDENTITY IS NOT FREE EITHER: a random basis picks head 5 as layer
#          10's top head in FEWER than 50% of the 60 trials. (If a random basis
#          picks head 5 most of the time, the identification is worth nothing --
#          the same failure mode membership had.)
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE_STOP = 11        # S1597 SITE: forward stops here, P = its input
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_grain_multiseed_results.json'
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
                 'published_top': ['attn10', 'attn9', 'mlp9', 'mlp10']},
}
NAMED = ['attn10', 'attn9']       # kept for reporting; the CLAIM here is head-grain
HEAD_LAYER = 10                   # S1597: attn10 -> head 10.5
HEAD_IDX = 5
N_RANDOM = 20                     # seeds 1729+i; i=0 IS S1612's own draw
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
    # per-head absolute mass within HEAD_LAYER, same statistic, same coefficient
    ch = coef[f'attn{HEAD_LAYER}']
    hmu = {h: acc['hsum'][HEAD_LAYER][h] / max(acc['n'], 1) for h in range(9)}
    hcmu = {h: acc['hcsum'][HEAD_LAYER][h] / max(acc['cn'], 1) for h in range(9)}
    hdelta = {h: (ch * (hcmu[h] - hmu[h])).abs().sum().item() for h in range(9)}
    return delta, acc['cn'], rec_err, hdelta


def head_stats(hdelta):
    """Top head index, top:next ratio (absolute mass within the layer)."""
    order = sorted(hdelta, key=lambda h: -hdelta[h])
    top, nxt = hdelta[order[0]], hdelta[order[1]]
    return order[0], (top / nxt if nxt > 0 else float('inf')), order


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
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {CHUNKS} x {ROWS_PER_CHUNK} rows x '
          f'{N_RANDOM} seeds | head-grain at layer {HEAD_LAYER} (receipt {rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    cfg = CELLS['question']; site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
    mask_v = rx(cfg['pat'])
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
    Dw = H[site].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:rank]
    V2l, lam2l = V[:, o].contiguous(), lam[o].contiguous()

    lam_top_head, lam_ratio, lam_hdelta = [], [], []
    for ci, ch in enumerate(chunks):
        delta, cn, rec, hd = abs_mass(ch, V2l, lam2l, mask_v)
        th, r, order = head_stats(hd)
        lam_top_head.append(th); lam_ratio.append(r)
        lam_hdelta.append({h: round(hd[h], 2) for h in range(9)})
        print(f'  lambda chunk{ci} n={cn:4d} top head {HEAD_LAYER}.{th} ratio {r:.2f} '
              f'order {order[:3]}', flush=True)

    rnd_ratio, rnd_tophead = [], []
    seed0 = []
    for i in range(N_RANDOM):
        gen = torch.Generator(device=DEV).manual_seed(1729 + i)
        rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
        for ci, ch in enumerate(chunks):
            delta, cn, rec, hd = abs_mass(ch, rnd.contiguous(),
                                          torch.ones(rank, device=DEV), mask_v)
            th, r, order = head_stats(hd)
            rnd_ratio.append(r); rnd_tophead.append(th)
            if i == 0:
                seed0.append((th, round(r, 2)))
        if i % 5 == 0:
            print(f'  random seed {1729+i} done ({i+1}/{N_RANDOM})', flush=True)

    finite = sorted(r for r in rnd_ratio if r != float('inf'))
    p95 = finite[int(0.95 * (len(finite) - 1))]
    frac_head = sum(1 for h in rnd_tophead if h == HEAD_IDX) / len(rnd_tophead)

    pa = all(h == HEAD_IDX for h in lam_top_head)
    pb = all(r > p95 for r in lam_ratio)
    pc = frac_head < 0.50

    print(f"\n  question@mlp11 r2, head-grain within layer {HEAD_LAYER}, "
          f"{N_RANDOM} bases x {CHUNKS} chunks = {len(rnd_ratio)} control trials", flush=True)
    print(f"    lambda top head   {[f'{HEAD_LAYER}.{h}' for h in lam_top_head]}", flush=True)
    print(f"    lambda ratio      {[round(r,2) for r in lam_ratio]}   "
          f"(§1597 published 20:1)", flush=True)
    print(f"    random ratio      min {finite[0]:.2f}  median {finite[len(finite)//2]:.2f}  "
          f"p95 {p95:.2f}  max {finite[-1]:.2f}", flush=True)
    print(f"    random picks head {HEAD_IDX} as top: "
          f"{sum(1 for h in rnd_tophead if h==HEAD_IDX)}/{len(rnd_tophead)} = {frac_head:.1%}",
          flush=True)
    print(f"    seed 1729 (top head, ratio) per chunk: {seed0}", flush=True)
    print(f"    lambda per-head mass chunk0: {lam_hdelta[0]}", flush=True)

    out = {'config': {'cell': 'question@mlp11 rank-2', 'head_layer': HEAD_LAYER,
                      'head_idx_claimed': HEAD_IDX, 'n_random': N_RANDOM,
                      'seeds': f'1729..{1729+N_RANDOM-1}', 'chunks': CHUNKS,
                      'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'statistic': 'per-head ABSOLUTE attribution mass within the layer, corrected quantity'},
           'lambda_top_head': lam_top_head, 'lambda_ratio': [round(r, 3) for r in lam_ratio],
           'lambda_head_mass': lam_hdelta,
           'random_ratio_min': round(finite[0], 3),
           'random_ratio_median': round(finite[len(finite)//2], 3),
           'random_ratio_p95': round(p95, 3), 'random_ratio_max': round(finite[-1], 3),
           'random_picks_claimed_head_fraction': round(frac_head, 4),
           'seed1729_per_chunk': seed0,
           'predictions': {'pred_a_lambda_top_head_is_claimed': bool(pa),
                           'pred_b_lambda_ratio_above_random_p95': bool(pb),
                           'pred_c_random_picks_head_lt_half': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
