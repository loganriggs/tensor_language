# snr_replication: PRE-REGISTERED TEST OF THE S1615 POST-HOC SNR CANDIDATE, ON
# DISJOINT CLASSES.
#
# S1615's registered hypothesis (null share set by deviation magnitude DEV) FAILED
# at rho .224, p .524. Looking at the data afterwards, a different quantity fit:
#     rho(DEV / shuffled_DEV, share) = .7333, p = .0201
# i.e. the SIGNAL-TO-NOISE ratio, beating DEV (.224), n (.673) and shuffled DEV
# alone (-.673). S1615 recorded it as POST-HOC and explicitly UNSCORED, because it
# was chosen from four candidates on the same 10 points after the registered bar
# failed -- a garden-of-forking-paths selection.
#
# This is the confirmatory run it requires. The 10 classes below are DISJOINT from
# the S1615 discovery set {question, months, days, semicolon, colon, pronouns, is,
# said, to, the} -- no overlap -- and span n = 115 to 7442, comparable to the
# discovery range (118-6911). Everything else is identical: random rank-2 basis,
# TOP 4, absolute-mass statistic, mlp11, local curated_rows.pt 3 x 333, seed 1729,
# matched shuffled-label control per class.
#
# Registered predictions:
#   pred_a SNR REPLICATES: rho(DEV/shuffled, share) >= .60 with 20k-permutation
#          2-sided p < .05 on these disjoint classes.
#   pred_b SNR BEATS n on the SAME disjoint data: |rho(SNR)| > |rho(n)|.
#   pred_c the S1615 NOISE-FLOOR LAW replicates: rho(1/sqrt(n), shuffled DEV)
#          >= .95 (S1615 measured exactly 1.000 on the discovery classes).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'snr_replication_results.json'
NR = 960
SITE = 11
RANK = 2
TOP = 4
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


CLASSES = {          # ALL DISJOINT from the S1615 discovery set
    'dollar':      r'^\$$|^ \$$',
    'about':       r'^ about$',
    'not':         r'^ not$',
    'close_paren': r'^\)$|^ \)$',
    'open_quote':  r'^"$|^ "$',
    'open_paren':  r'^\($|^ \($',
    'with':        r'^ with$',
    'that':        r'^ that$',
    'hyphen':      r'^-$|^ -$',
    'comma':       r'^,$|^ ,$',
}
CHUNKS, ROWS_PER_CHUNK = 3, 333


@torch.no_grad()
def measure(rows, V2, mask_v, shuffle_gen=None):
    """Returns (top-TOP share, total attribution mass DEV, class_n).
    If shuffle_gen is given, the class mask is REPLACED by a random mask of the
    same size drawn from the valid positions -- the matched shuffled-label control."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    rk = V2.shape[1]
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    lam2 = torch.ones(rk, device=DEV)
    masks = []
    for i in range(0, rows.shape[0], 8):
        tg = rows[i:i + 8, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        masks.append(pm)
    if shuffle_gen is not None:
        flat = torch.cat([p.reshape(-1) for p in masks])
        valid = torch.zeros_like(flat)
        off = 0
        for p in masks:                       # valid = positions >= 64, any token
            v = torch.ones_like(p); v[:, :64] = False
            valid[off:off + p.numel()] = v.reshape(-1); off += p.numel()
        k = int(flat.sum())
        vidx = valid.nonzero(as_tuple=True)[0]
        pick = vidx[torch.randperm(vidx.numel(), device=DEV, generator=shuffle_gen)[:k]]
        newflat = torch.zeros_like(flat); newflat[pick] = True
        off = 0
        for j, p in enumerate(masks):
            masks[j] = newflat[off:off + p.numel()].reshape(p.shape); off += p.numel()
    for j, i in enumerate(range(0, rows.shape[0], 8)):
        capture_fwd(rows[i:i + 8, :-1].to(DEV).contiguous(), V2, lam2, acc, masks[j])

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
    return sum(delta[c] for c in ranked[:TOP]) / max(tot, 1e-9), tot, acc['cn']


def spear(x, y, trials=20000, seed=SEED):
    import random
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(o): r[i] = p + 1
        return r
    rx, ry = rk(x), rk(y); N = len(x)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (N * (N * N - 1))
    rnd = random.Random(seed); base = list(range(1, N + 1)); cnt = 0
    for _ in range(trials):
        pm = base[:]; rnd.shuffle(pm)
        r = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, pm)) / (N * (N * N - 1))
        cnt += (abs(r) >= abs(rho))
    return round(rho, 4), round(cnt / trials, 4)


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    g = torch.Generator(device=DEV).manual_seed(SEED)
    V2, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=g)); V2 = V2.contiguous()
    masks = {k: rx(v) for k, v in CLASSES.items()}
    names = list(CLASSES)

    out_c = {}
    for cname in names:
        sh, dv, ns = [], [], []
        for ch in chunks:
            s_, d_, n_ = measure(ch, V2, masks[cname])
            sh.append(s_); dv.append(d_); ns.append(n_)
        sg = torch.Generator(device=DEV).manual_seed(SEED)
        sdv = []
        for ch in chunks:
            _, d2, _ = measure(ch, V2, masks[cname], shuffle_gen=sg)
            sdv.append(d2)
        out_c[cname] = {'share': round(sum(sh) / 3, 4), 'dev': round(sum(dv) / 3, 4),
                        'dev_shuffled': round(sum(sdv) / 3, 4), 'n': sum(ns)}
        r = out_c[cname]
        print(f"  {cname:10s} n={r['n']:5d} share={r['share']:.4f} DEV={r['dev']:9.2f} "
              f"shuf={r['dev_shuffled']:9.2f} ratio={r['dev']/max(r['dev_shuffled'],1e-9):6.2f}x", flush=True)

    import math
    share = [out_c[c]['share'] for c in names]
    dev   = [out_c[c]['dev'] for c in names]
    shuf  = [out_c[c]['dev_shuffled'] for c in names]
    nn    = [out_c[c]['n'] for c in names]
    snr   = [out_c[c]['dev'] / max(out_c[c]['dev_shuffled'], 1e-9) for c in names]

    rho_snr, p_snr = spear(snr, share)
    rho_n,   p_n   = spear(nn, share)
    rho_dev, p_dev = spear(dev, share)
    rho_floor, p_floor = spear([1.0 / math.sqrt(x) for x in nn], shuf)

    pa = rho_snr >= 0.60 and p_snr < 0.05
    pb = abs(rho_snr) > abs(rho_n)
    pc = rho_floor >= 0.95

    print(f"\n  rho(SNR, share)      = {rho_snr} p={p_snr}   (S1615 post-hoc .7333)", flush=True)
    print(f"  rho(n, share)        = {rho_n} p={p_n}   (S1614 .6727)", flush=True)
    print(f"  rho(DEV, share)      = {rho_dev} p={p_dev}   (S1615 registered .2242, FAILED)", flush=True)
    print(f"  rho(1/sqrt n, shuf)  = {rho_floor} p={p_floor}   (S1615 law 1.000)", flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'top': TOP, 'seed': SEED,
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK,
                      'row_source': 'curated_rows.pt', 'rows_are_fresh': False,
                      'classes': names, 'arm': 'RANDOM basis only'},
           'per_class': out_c,
           'spearman': {'snr_vs_share': {'rho': rho_snr, 'p': p_snr},
                        'n_vs_share': {'rho': rho_n, 'p': p_n},
                        'dev_vs_share': {'rho': rho_dev, 'p': p_dev},
                        'inv_sqrt_n_vs_shuffled': {'rho': rho_floor, 'p': p_floor}},
           'discovery_reference': {'S1615_snr_posthoc': 0.7333, 'S1614_n': 0.6727,
                                   'S1615_dev_registered': 0.2242, 'S1615_floor_law': 1.000},
           'classes_dev_above_shuffled': beats,
           'predictions': {'pred_a_snr_replicates_ge60': bool(pa),
                           'pred_b_snr_beats_n': bool(pb),
                           'pred_c_noise_floor_law_ge95': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  rho(DEV, share)={rho_dev} p={p_dev} | rho(n, share)={rho_n} p={p_n} "
          f"(S1614 n rho .6727)", flush=True)
    print(f"  DEV above shuffled control in {beats}/10 classes", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
