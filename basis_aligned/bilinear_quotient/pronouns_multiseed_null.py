# pronouns_multiseed_null: DOES S1598/S1624's "DIFFUSE" VERDICT SURVIVE A PROPER
# MULTI-SEED NULL, OR WAS SEED 1729 FLATTERING IT?
#
# S1598's .482 was called a failure of sparsity; S1612 reinterpreted it as a
# POSITIVE structural claim (the slice is DRAMATICALLY less concentrated than a
# meaningless basis) and S1624 confirmed .4823 against a null of .5744. Every one
# of those nulls comes from a SINGLE random basis, seed 1729.
#
# S1628 just showed that seed 1729 is NOT representative at the other cell: for
# question@mlp11 it gave a null of .586-.612 where the 20-seed mean is .567-.573,
# i.e. it ran HIGH by about .03. If seed 1729 also runs high at mlp17, the pronouns
# null is really LOWER than .5744, the below-null gap SHRINKS, and the "dramatically
# diffuse" reading weakens or reverses. That is a live risk to a published reading,
# so it gets measured rather than assumed.
#
# Design: pronouns@mlp17 rank-8 TOP-6, corrected quantity (forward stops at 17,
# upstream comps, site-relative coefficients), 20 INDEPENDENT random rank-8 bases
# (seeds 1729-1748, so S1612/S1624's own draw is i=0 and is reported separately)
# x 3 DISJOINT 160-row chunks = 60 trials, on canonical fineweb_n480_skip80.pt.
#
# Registered predictions:
#   pred_a THE DIFFUSE VERDICT SURVIVES: the lambda share is BELOW the 20-seed mean
#          null on all three chunks.
#   pred_b SEED 1729 IS UNREPRESENTATIVE HERE TOO: the 20-seed mean null differs
#          from S1624's single-seed .5744 by >= .01 in absolute value.
#   pred_c THE SEPARATION IS PERFECT DOWNWARD, mirroring S1628's perfect upward
#          separation at question: the lambda arm's HIGHEST chunk share is below the
#          MINIMUM of all 60 random trial shares.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE_STOP = 17        # S1598 SITE: forward stops here, P = its input
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pronouns_multiseed_null_results.json'
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
    'pronouns': {'site': 17, 'rank': 8, 'top': 6,
                 'pat': r'^ (he|she|they|He|She|They)$', 'published': 0.482,
                 'published_top': ['mlp16', 'x0', 'mlp15', 'mlp9', 'mlp14', 'attn9']},
}
NAMED = ['mlp16', 'x0']           # S1598's two largest published writers
N_RANDOM = 20                     # seeds 1729+i; i=0 IS S1612's own draw
CHUNKS, ROWS_PER_CHUNK = 3, 160        # three DISJOINT chunks
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT  = PT + '.rowcache/fineweb_oracle_v2_receipt.json'


@torch.no_grad()
def abs_mass(rows, V2, lam2, mask_v):
    """Per-component ABSOLUTE attribution mass -- the S1597/S1598 statistic:
    delta_c = |coef_c * (class_mean - global_mean)| summed over slice directions."""
    SITE_UP = 17          # S1598: comps are UPSTREAM of the slice site
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
    import os, hashlib
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    assert raw.shape[0] >= CHUNKS * ROWS_PER_CHUNK, f'short row tensor {tuple(raw.shape)}'
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {CHUNKS} chunks x {ROWS_PER_CHUNK} rows '
          f'(receipt sha256[:16]={rh})', flush=True)
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    cfg = CELLS['pronouns']; site, rank, TOP = cfg['site'], cfg['rank'], cfg['top']
    mask_v = rx(cfg['pat'])
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
    Dw = H[site].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:rank]
    V2l, lam2l = V[:, o].contiguous(), lam[o].contiguous()

    lam_share, lam_top = [], []
    for ci, ch in enumerate(chunks):
        delta, cn, rec = abs_mass(ch, V2l, lam2l, mask_v)
        ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
        top = ranked[:TOP]; sh = sum(delta[c] for c in top) / max(tot, 1e-9)
        lam_share.append(sh); lam_top.append(top)
        print(f'  lambda chunk{ci} n={cn:4d} share={sh:.4f} top={top} recon={rec:.1e}', flush=True)

    # 20 INDEPENDENT random bases x 3 chunks
    rnd_share = [[] for _ in range(CHUNKS)]
    present = {c: 0 for c in NAMED}; trials = 0
    seed0 = {c: [] for c in NAMED}          # S1612's own draw, reported separately
    per_seed = []
    for i in range(N_RANDOM):
        gen = torch.Generator(device=DEV).manual_seed(1729 + i)
        rnd, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
        rec_i = {'seed': 1729 + i, 'chunks': []}
        for ci, ch in enumerate(chunks):
            delta, cn, rec = abs_mass(ch, rnd.contiguous(), torch.ones(rank, device=DEV), mask_v)
            ranked = sorted(delta, key=lambda c: -delta[c]); tot = sum(delta.values())
            top = ranked[:TOP]; sh = sum(delta[c] for c in top) / max(tot, 1e-9)
            rnd_share[ci].append(sh); trials += 1
            for c in NAMED:
                if c in top:
                    present[c] += 1
                    if i == 0:
                        seed0[c].append(ci)
            rec_i['chunks'].append({'chunk': ci, 'share': round(sh, 4), 'top': top})
        per_seed.append(rec_i)
        if i % 5 == 0:
            print(f'  random seed {1729+i} done ({i+1}/{N_RANDOM})', flush=True)

    mean_rnd = [sum(v) / len(v) for v in rnd_share]
    gaps = [round(l - r, 4) for l, r in zip(lam_share, mean_rnd)]
    frac = {c: present[c] / trials for c in NAMED}

    all_rnd = [c['share'] for r in per_seed for c in r['chunks']]
    mean_null_overall = sum(all_rnd) / len(all_rnd)
    S1624_SEED_NULL = 0.5744
    pa = all(l < r for l, r in zip(lam_share, mean_rnd))          # still BELOW its null
    pb = abs(mean_null_overall - S1624_SEED_NULL) >= 0.01          # seed 1729 unrepresentative
    pc = max(lam_share) < min(all_rnd)                             # perfect downward separation

    print(f"\n  pronouns@mlp17 rank-8 TOP-6, {N_RANDOM} random bases x {CHUNKS} chunks "
          f"= {trials} trials", flush=True)
    for c in NAMED:
        print(f"    {c:7s} in random top-6: {present[c]}/{trials} = {frac[c]:.1%}"
              f"   | seed 1729: chunks {seed0[c] if seed0[c] else 'NONE'}", flush=True)
    print(f"  20-seed mean null overall {mean_null_overall:.4f}  vs S1624 seed-1729 "
          f"null {S1624_SEED_NULL}  (delta {mean_null_overall - S1624_SEED_NULL:+.4f})", flush=True)
    print(f"  60-trial random share range {min(all_rnd):.4f} - {max(all_rnd):.4f}"
          f"   | lambda max {max(lam_share):.4f}", flush=True)
    print(f"  lambda share  {[round(x,4) for x in lam_share]}", flush=True)
    print(f"  mean random   {[round(x,4) for x in mean_rnd]}", flush=True)
    print(f"  gap           {gaps}", flush=True)
    print(f"  lambda top-4 per chunk: {lam_top}", flush=True)

    out = {'config': {'cell': 'pronouns@mlp17 rank-8 TOP-6', 'named': NAMED,
                      'n_random': N_RANDOM, 'seeds': f'1729..{1729+N_RANDOM-1}',
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority pinned_local_ordered_manifest)',
                      'statistic': 'ABSOLUTE attribution mass, corrected quantity (forward stops at site, upstream comps, site-relative coefs)'},
           'lambda_share': [round(x, 4) for x in lam_share], 'lambda_top': lam_top,
           'mean_random_share': [round(x, 4) for x in mean_rnd], 'gaps': gaps,
           'present_in_random_top4': present, 'trials': trials,
           'fraction': {c: round(frac[c], 4) for c in NAMED},
           'seed1729_chunks_present': seed0,
           'per_seed': per_seed,
           'predictions': {'pred_a_lambda_below_null_all_chunks': bool(pa),
                           'pred_b_seed1729_unrepresentative_ge_01': bool(pb),
                           'pred_c_perfect_downward_separation': bool(pc)},
           'mean_null_overall': round(mean_null_overall, 4),
           'S1624_seed_null': S1624_SEED_NULL,
           'random_share_range': [round(min(all_rnd), 4), round(max(all_rnd), 4)],
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
