# headgrain_normcorrect: CAN THE HEAD-GRAIN STATISTIC BE FLOOR-CORRECTED?
#
# S1617: the head-grain law generalises at 40% over 30 cells vs 13.3% random, but
# only 26.7% after subtracting cells where the random arm finds the same head.
# S1618 then showed CAUSALLY why it misses: at digits L7 and L12 the |lambda|
# slice picks heads with selectivity 0.24 and 0.10 -- BELOW 1, so they damage
# global function more than the class. They are high-norm GENERALISTS. The
# head-grain score is |coef * (cmu - mu)| projected through c_proj, so a head with
# a large output norm scores highly whatever it computes.
#
# Obvious correction: divide each head's score by its c_proj slice norm, so the
# statistic measures DIRECTION agreement rather than magnitude. If the floor is a
# norm artifact, this should raise the hit rate on signal without raising it on
# the random arm.
#
# Same 30 cells as S1617 -- 6 classes (colon, months, close_paren, digits, comma,
# and) x mlp11 and mlp17 x their certified layers -- both statistics computed in
# ONE pass from the same accumulators, so raw and corrected are exactly comparable.
# |lambda|-r8 vs matched-rank random, local curated_rows.pt 3 x 333, seed 1729.
#
# Registered predictions:
#   pred_a NORM-CORRECTION HELPS SIGNAL: the corrected |lambda| hit rate is >= 50%
#          (S1617's raw rate was 40.0%).
#   pred_b IT DOES NOT HELP NOISE: the corrected RANDOM hit rate stays <= 20%
#          (raw was 13.3%). A correction that lifts both is measuring nothing.
#   pred_c THE digits CELLS IMPROVE: >= 2 of the 8 digits cells become hits under
#          correction (S1617 raw: 0/8), since S1618 showed their misses are
#          precisely the high-norm generalists this correction targets.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'headgrain_normcorrect_results.json'
NR = 960
RANK = 8
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




CERTIFIED = {
    'colon':       ['12.6'],
    'months':      ['14.7'],
    'close_paren': ['13.8'],
    'digits':      ['7.3', '6.5', '12.6', '11.5'],
    'comma':       ['6.5', '15.2', '9.5', '11.7'],
    'and':         ['10.5', '16.8', '7.3', '9.5'],
}
PATTERNS = {
    'colon':       r'^:$|^ :$',
    'months':      r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$',
    'close_paren': r'^\)$|^ \)$',
    'digits':      r'^ ?[0-9]+$',
    'comma':       r'^,$|^ ,$',
    'and':         r'^ and$',
}
SITES = [11, 17]
CHUNKS, ROWS_PER_CHUNK = 3, 333


def head_norms():
    """||c_proj slice|| per (layer, head) -- the quantity the correction divides by."""
    out = {}
    for L in range(18):
        W = H[L].attn.c_proj.weight.float()
        for h in range(9):
            out[(L, h)] = float(W[:, h * 128:(h + 1) * 128].norm())
    return out


@torch.no_grad()
def grain_both(rows, V2, mask_v, hn):
    """Returns {layer: {'raw': top_head, 'corr': top_head}} -- the S1597/S1598
    head-grain statistic and its norm-corrected form, from ONE pass."""
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
        tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
        capture_fwd(bb[:, :-1].to(DEV).contiguous(), V2, lam2, acc, pm)
    lam0 = [float(b.lambdas[0]) for b in H]
    coef = {}
    for l in range(18):
        c = 1.0
        for kk in range(l + 1, 18):
            c *= lam0[kk]
        coef[f'attn{l}'] = c
    out = {}
    for L in range(18):
        raw = {h: float((coef[f'attn{L}'] * (acc['hcsum'][L][h] / max(acc['cn'], 1)
                         - acc['hsum'][L][h] / max(acc['n'], 1))).abs().sum())
               for h in range(9)}
        corr = {h: raw[h] / max(hn[(L, h)], 1e-9) for h in range(9)}
        out[L] = {'raw': f"{L}.{max(raw, key=raw.get)}",
                  'corr': f"{L}.{max(corr, key=corr.get)}"}
    return out


@torch.no_grad()
def main():
    import os
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw_rows = torch.load(PT + 'curated_rows.pt', map_location='cpu')['rows']
    allr = raw_rows[:CHUNKS * ROWS_PER_CHUNK, :T + 1].contiguous()
    chunks = [allr[c * ROWS_PER_CHUNK:(c + 1) * ROWS_PER_CHUNK] for c in range(CHUNKS)]
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    hn = head_norms()
    nv = [hn[(L, h)] for L in range(18) for h in range(9)]
    print(f"head c_proj norms: min {min(nv):.3f} max {max(nv):.3f} ratio {max(nv)/min(nv):.2f}x",
          flush=True)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    rnd, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=g)); rnd = rnd.contiguous()

    cells = []
    for cname, heads in CERTIFIED.items():
        mask_v = rx(PATTERNS[cname])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        for site in SITES:
            Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
            Dw = H[site].mlp.Down.weight.float()
            Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
            S = 0.5 * (Q + Q.T)
            lam, V = torch.linalg.eigh(S)
            o = lam.abs().argsort(descending=True)[:RANK]
            arms = {'lambda': V[:, o].contiguous(), 'random': rnd}
            got = {a: [grain_both(ch, V2, mask_v, hn) for ch in chunks]
                   for a, V2 in arms.items()}
            for hd in heads:
                L = int(hd.split('.')[0])
                rec = {'cell': f'{cname}@mlp{site} L{L}', 'class': cname,
                       'certified': hd}
                for a in ('lambda', 'random'):
                    for stat in ('raw', 'corr'):
                        tops = [got[a][i][L][stat] for i in range(CHUNKS)]
                        rec[f'{a}_{stat}_tops'] = tops
                        rec[f'{a}_{stat}_hit'] = sum(1 for t in tops if t == hd) >= 2
                cells.append(rec)
                print(f"  {rec['cell']:22s} cert={hd:6s} "
                      f"raw={'HIT ' if rec['lambda_raw_hit'] else '    '}{rec['lambda_raw_tops'][0]:6s} "
                      f"corr={'HIT ' if rec['lambda_corr_hit'] else '    '}{rec['lambda_corr_tops'][0]:6s} "
                      f"rnd_corr={'hit' if rec['random_corr_hit'] else '   '}", flush=True)

    N = len(cells)
    def rate(k): return sum(1 for c in cells if c[k]) / N
    lr, lc = rate('lambda_raw_hit'), rate('lambda_corr_hit')
    rr, rc = rate('random_raw_hit'), rate('random_corr_hit')
    dig = [c for c in cells if c['class'] == 'digits']
    dig_corr = sum(1 for c in dig if c['lambda_corr_hit'])

    pa = lc >= 0.50
    pb = rc <= 0.20
    pc = dig_corr >= 2

    out = {'config': {'classes': list(CERTIFIED), 'sites': SITES, 'rank': RANK,
                      'chunks': CHUNKS, 'rows_per_chunk': ROWS_PER_CHUNK, 'seed': SEED,
                      'row_source': 'curated_rows.pt', 'rows_are_fresh': False,
                      'correction': 'head score / ||c_proj slice||'},
           'cells': cells, 'n_cells': N,
           'rates': {'lambda_raw': round(lr, 4), 'lambda_corrected': round(lc, 4),
                     'random_raw': round(rr, 4), 'random_corrected': round(rc, 4)},
           'digits_corrected_hits': dig_corr, 'digits_cells': len(dig),
           'S1617_reference': {'lambda_raw': 0.40, 'random_raw': 0.133},
           'predictions': {'pred_a_corrected_lambda_ge50': bool(pa),
                           'pred_b_corrected_random_le20': bool(pb),
                           'pred_c_digits_ge2_hits': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\n  lambda raw {lr:.1%} -> corrected {lc:.1%}   |   random raw {rr:.1%} -> corrected {rc:.1%}",
          flush=True)
    print(f"  digits corrected hits {dig_corr}/{len(dig)} (S1617 raw 0/8)", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
