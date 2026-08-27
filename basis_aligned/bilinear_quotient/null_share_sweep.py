# null_share_sweep: WHAT DETERMINES THE MATCHED-RANK NULL SHARE?
# S1612 showed the null is CELL-DEPENDENT -- .4489 at question@mlp11 (rank-2,
# TOP=4) and .7295 at pronouns@mlp17 (rank-8, TOP=6) -- so a top-K share can only
# be read against its own null, and every future writer claim currently needs its
# own random arm. If the null is predictable from (rank, TOP) alone and does NOT
# depend on the class, it can be looked up instead of re-measured.
#
# Two sweeps on the ABSOLUTE-mass statistic (slice_writers.py:216), random arm
# only, local curated_rows.pt, 3 disjoint 333-row chunks:
#   A. class sweep  -- site mlp11, rank 2, TOP 4, classes {question, pronouns, the, is}
#   B. rank  sweep  -- site mlp11, class question, TOP 4, ranks {2, 4, 8, 16}
# The random basis is drawn from the SAME seed per (rank) so arms are comparable.
# Note the null depends only on the random subspace and the residual stream, not
# on the class-projected form -- so a class effect, if any, enters ONLY through
# the class MASK selecting different positions.
#
# Registered predictions:
#   pred_a CLASS-INDEPENDENT: across the 4 classes at fixed (mlp11, rank 2,
#          TOP 4), the null share range (max - min) is <= .05.
#   pred_b MONOTONE IN RANK: across ranks {2, 4, 8, 16} at fixed class, the null
#          share is monotone (non-increasing or non-decreasing throughout).
#   pred_c SELF-CONSISTENT: the question@mlp11 rank-2 TOP-4 null reproduces
#          S1612's .4489 within .05.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'null_share_sweep_results.json'
NR = 960
SITE = 11          # fixed for both sweeps
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


CLASSES = {'question': r'^\?$| \?$',
           'pronouns': r'^ (he|she|they|He|She|They)$',
           'the': r'^ the$',
           'is': r'^ is$'}
CLASS_SWEEP = ['question', 'pronouns', 'the', 'is']
RANK_SWEEP = [2, 4, 8, 16]
TOP = 4
CHUNKS, ROWS_PER_CHUNK = 3, 333


@torch.no_grad()
def null_share(rows, V2, mask_v, top):
    """Absolute-mass top-K share for a RANDOM basis -- the S1597/S1598 statistic."""
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
    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    delta = {c: (coef[c] * (cmu[c] - mu[c])).abs().sum().item() for c in comps}
    ranked = sorted(delta, key=lambda c: -delta[c])
    tot = sum(delta.values())
    return sum(delta[c] for c in ranked[:top]) / max(tot, 1e-9), acc['cn']


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    masks = {k: rx(v) for k, v in CLASSES.items()}

    def rand_basis(rank):
        g = torch.Generator(device=DEV).manual_seed(1729)
        Q, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=g))
        return Q.contiguous()

    print('=== SWEEP A: class, at mlp11 rank-2 TOP-4 ===', flush=True)
    V2 = rand_basis(2); classA = {}
    for cname in CLASS_SWEEP:
        vals = []
        for ch in chunks:
            sh, cn = null_share(ch, V2, masks[cname], TOP)
            vals.append(round(sh, 4))
        classA[cname] = {'shares': vals, 'mean': round(sum(vals) / len(vals), 4), 'last_n': cn}
        print(f"  {cname:9s} null shares {vals} mean {classA[cname]['mean']:.4f} (n={cn})", flush=True)
    means = [classA[c]['mean'] for c in CLASS_SWEEP]
    rng = round(max(means) - min(means), 4)

    print('\n=== SWEEP B: rank, at mlp11 question TOP-4 ===', flush=True)
    rankB = {}
    for rank in RANK_SWEEP:
        Vr = rand_basis(rank); vals = []
        for ch in chunks:
            sh, cn = null_share(ch, Vr, masks['question'], TOP)
            vals.append(round(sh, 4))
        rankB[rank] = {'shares': vals, 'mean': round(sum(vals) / len(vals), 4)}
        print(f"  rank {rank:2d}  null shares {vals} mean {rankB[rank]['mean']:.4f}", flush=True)
    rmeans = [rankB[r]['mean'] for r in RANK_SWEEP]
    inc = all(rmeans[i] <= rmeans[i + 1] for i in range(len(rmeans) - 1))
    dec = all(rmeans[i] >= rmeans[i + 1] for i in range(len(rmeans) - 1))

    pa = rng <= 0.05
    pb = inc or dec
    pc = abs(rankB[2]['mean'] - 0.4489) <= 0.05

    out = {'config': {'site': SITE, 'top': TOP, 'chunks': CHUNKS,
                      'rows_per_chunk': ROWS_PER_CHUNK, 'row_source': 'curated_rows.pt',
                      'rows_are_fresh': False, 'arm': 'RANDOM only',
                      'statistic': 'absolute attribution mass (slice_writers.py:216)'},
           'class_sweep': classA, 'class_mean_range': rng,
           'rank_sweep': {str(k): v for k, v in rankB.items()},
           'rank_monotone': {'increasing': inc, 'decreasing': dec},
           'S1612_reference_null': 0.4489,
           'predictions': {'pred_a_class_independent_range_le05': bool(pa),
                           'pred_b_monotone_in_rank': bool(pb),
                           'pred_c_reproduces_S1612_null_within_05': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\nclass-mean range {rng} | rank means {rmeans} inc={inc} dec={dec}", flush=True)
    print(f"rank-2 null {rankB[2]['mean']:.4f} vs S1612's .4489", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
