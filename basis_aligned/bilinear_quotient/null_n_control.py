# null_n_control: DOES CLASS SAMPLE SIZE EXPLAIN THE NULL SHARE?
# S1613 found the matched-rank null share is CLASS-dependent (range .1417 across
# 4 classes, 3.3x the rank effect) and noted a loose association with class
# count -- question (n=102) lowest at .4489, 'the' (n=2354) highest at .5906 --
# but tested it and found rho = .800 at exact permutation p = .167 on N=4.
# UNDERPOWERED, not supported. This tests it at power, and then controls it.
#
# If class-n IS the mechanism, two things follow and both are serious:
#   * low-n classes show artificially LOW apparent concentration, so a slice at a
#     rare class looks more "diffuse" than one at a common class for reasons that
#     have nothing to do with mechanism;
#   * cross-class share comparisons anywhere in this program are biased, which
#     directly affects S1598's .482 being read against S1597's .718.
#
# TWO LEGS, random arm only, absolute-mass statistic (slice_writers.py:216),
# local curated_rows.pt, 3 x 333 rows, site mlp11, rank 2, TOP 4:
#   A. POWERED: null share for 10 classes spanning ~10x in n; Spearman rho vs n
#      with an EXACT permutation p-value.
#   B. n-CONTROLLED (the decisive leg): repeat with every class SUBSAMPLED to a
#      common budget of class positions (the minimum across classes), same RNG
#      seed. If n is the mechanism, the class range must collapse.
#
# Registered predictions:
#   pred_a POWERED: rho(class n, null share) >= .60 with exact permutation
#          p < .05 over the 10 classes.
#   pred_b CONTROLLED: with n equalised, the class-mean null range falls below
#          .05 (i.e. n explains the S1613 class effect).
#   pred_c the control does not simply flatten everything: the equalised nulls
#          still vary by >= .01, so the subsampling has not destroyed signal.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'null_n_control_results.json'
NR = 960
SITE = 11
RANK = 2
TOP = 4
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


CLASSES = {
    'question':    r'^\?$| \?$',
    'months':      r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$',
    'days':        r'^ (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
    'semicolon':   r'^;$|^ ;$',
    'colon':       r'^:$|^ :$',
    'pronouns':    r'^ (he|she|they|He|She|They)$',
    'is':          r'^ is$',
    'said':        r'^ said$',
    'to':          r'^ to$',
    'the':         r'^ the$',
}
CHUNKS, ROWS_PER_CHUNK = 3, 333
SEED = 1729


@torch.no_grad()
def null_share(rows, V2, mask_v, budget=None, gen=None):
    """Absolute-mass top-TOP share for a RANDOM basis.
    If `budget` is given, the class-position mask is randomly SUBSAMPLED to that
    many positions per chunk, so classes can be compared at equal n."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    rk = V2.shape[1]
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    lam2 = torch.ones(rk, device=DEV)
    # build the full class mask first so the subsample budget is applied per CHUNK
    full = []
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        full.append(pm)
    if budget is not None:
        flat = torch.cat([p.reshape(-1) for p in full])
        idx = flat.nonzero(as_tuple=True)[0]
        if idx.numel() > budget:
            perm = torch.randperm(idx.numel(), device=DEV, generator=gen)[:budget]
            keep = torch.zeros_like(flat); keep[idx[perm]] = True
        else:
            keep = flat.clone()
        off = 0
        for k, p in enumerate(full):
            nn = p.numel()
            full[k] = keep[off:off + nn].reshape(p.shape)
            off += nn
    for k, i in enumerate(range(0, rows.shape[0], 8)):
        bb = rows[i:i + 8]
        idx_t = bb[:, :-1].to(DEV).contiguous()
        capture_fwd(idx_t, V2, lam2, acc, full[k])

    lam0 = [float(b.lambdas[0]) for b in H]; lam1 = [float(b.lambdas[1]) for b in H]
    coef = {}
    for l in range(18):
        c = 1.0
        for kk in range(l + 1, 18):
            c *= lam0[kk]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for kk in range(18):
        tx0 = lam0[kk] * tx0 + lam1[kk]
    coef['x0'] = tx0
    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    delta = {c: (coef[c] * (cmu[c] - mu[c])).abs().sum().item() for c in comps}
    ranked = sorted(delta, key=lambda c: -delta[c])
    tot = sum(delta.values())
    return sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9), acc['cn']


def spearman_exact(x, y):
    import itertools
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, i in enumerate(o): r[i] = pos + 1
        return r
    rx, ry = rank(x), rank(y); N = len(x)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (N * (N * N - 1))
    if N <= 8:
        tot = cnt = 0
        for perm in itertools.permutations(range(1, N + 1)):
            r = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, perm)) / (N * (N * N - 1))
            tot += 1; cnt += (r >= rho)
        return rho, cnt / tot, 'exact'
    # N>8: Monte-Carlo permutation, deterministic seed
    import random
    rnd = random.Random(SEED); tot = 20000; cnt = 0
    base = list(range(1, N + 1))
    for _ in range(tot):
        perm = base[:]; rnd.shuffle(perm)
        r = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, perm)) / (N * (N * N - 1))
        cnt += (r >= rho)
    return rho, cnt / tot, f'monte-carlo n={tot}'


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    g = torch.Generator(device=DEV).manual_seed(SEED)
    V2, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=g))
    V2 = V2.contiguous()
    masks = {k: rx(v) for k, v in CLASSES.items()}

    print('=== LEG A: powered, natural n ===', flush=True)
    legA = {}
    for cname in CLASSES:
        vals, ns = [], []
        for ch in chunks:
            sh, cn = null_share(ch, V2, masks[cname])
            vals.append(round(sh, 4)); ns.append(cn)
        legA[cname] = {'shares': vals, 'mean': round(sum(vals) / len(vals), 4),
                       'n_per_chunk': ns, 'n_total': sum(ns)}
        print(f"  {cname:10s} n={sum(ns):5d}  null {vals} mean {legA[cname]['mean']:.4f}", flush=True)

    names = list(CLASSES)
    ns = [legA[c]['n_total'] for c in names]
    sh = [legA[c]['mean'] for c in names]
    rho, pval, method = spearman_exact(ns, sh)
    rangeA = round(max(sh) - min(sh), 4)
    print(f"\n  rho(n, null) = {rho:.3f}  p = {pval:.4f} ({method})  range {rangeA}", flush=True)

    budget = max(1, min(legA[c]['n_total'] for c in names) // CHUNKS)
    print(f"\n=== LEG B: n-CONTROLLED, budget {budget} class positions per chunk ===", flush=True)
    legB = {}
    for cname in names:
        gb = torch.Generator(device=DEV).manual_seed(SEED)
        vals, ns2 = [], []
        for ch in chunks:
            sh2, cn2 = null_share(ch, V2, masks[cname], budget=budget, gen=gb)
            vals.append(round(sh2, 4)); ns2.append(cn2)
        legB[cname] = {'shares': vals, 'mean': round(sum(vals) / len(vals), 4), 'n_per_chunk': ns2}
        print(f"  {cname:10s} n={sum(ns2):5d}  null {vals} mean {legB[cname]['mean']:.4f}", flush=True)
    shB = [legB[c]['mean'] for c in names]
    rangeB = round(max(shB) - min(shB), 4)
    rhoB, pB, mB = spearman_exact([legB[c]['n_per_chunk'][0] for c in names], shB)

    pa = rho >= 0.60 and pval < 0.05
    pb = rangeB < 0.05
    pc = rangeB >= 0.01

    out = {'config': {'site': SITE, 'rank': RANK, 'top': TOP, 'chunks': CHUNKS,
                      'rows_per_chunk': ROWS_PER_CHUNK, 'seed': SEED,
                      'row_source': 'curated_rows.pt', 'rows_are_fresh': False,
                      'classes': names, 'arm': 'RANDOM only',
                      'statistic': 'absolute attribution mass (slice_writers.py:216)'},
           'leg_a_natural_n': legA, 'leg_b_n_controlled': legB,
           'budget_per_chunk': budget,
           'spearman_natural': {'rho': round(rho, 4), 'p': round(pval, 4), 'method': method},
           'range_natural': rangeA, 'range_controlled': rangeB,
           'spearman_controlled': {'rho': round(rhoB, 4), 'p': round(pB, 4)},
           'S1613_reference': {'4class_rho': 0.800, '4class_p': 0.167, '4class_range': 0.1417},
           'predictions': {'pred_a_powered_rho_ge60_p_lt05': bool(pa),
                           'pred_b_controlled_range_lt05': bool(pb),
                           'pred_c_controlled_range_ge01': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  natural range {rangeA} -> controlled range {rangeB}", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
