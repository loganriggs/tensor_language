# channel_depth_p: PRONOUNS CONFIRMATION of the channel-depth profile (S1602/03).
# S1603 certified, for the question class, that the mlp11-form channel accumulates
# MONOTONICALLY to a peak at attn16 (6/6 samples) and is then cut back .358 +/- .029
# by mlp16/mlp17, with mlp17 the largest single step of any sign. Second-class
# confirmation is this program's standing pattern (S1595, S1598).
#
# Pronouns differ structurally: the certified site is mlp17, not mlp11, and S1598
# recorded the rule that mlp17 slices must be SIGN-SPLIT (payload = the positive
# eigenpart), never |lambda|-ordered -- the |lambda|-top-8 slice there is
# suppression-dominated (6 of 8 eigs negative). So this uses pos_r8: the 8 MOST
# POSITIVE eigendirections of the class-projected form at mlp17.
# S1598 also died twice on hardcoded rank-2 accumulator shapes; RANK is threaded
# through every accumulator here.
# Six disjoint 96-row samples, spread quoted, no refitting. positions >=64,
# target-side class mask, pronouns = ' he|she|they|He|She|They'.
#
# Registered predictions:
#   pred_a the cumulative curve peaks at or AFTER layer 14 in >= 5 of 6 samples
#          (S1598 found pronoun writers are late-MLP: mlp16, mlp15, mlp14).
#   pred_b mlp17 is among the top-2 POSITIVE writers of its own form's payload
#          subspace, in >= 5 of 6 samples -- i.e. the S1601 read-write coincidence
#          (mlp11 was the largest writer of the subspace mlp11 reads) replicates
#          on a second class at a different site.
#   pred_c mean drawdown <= .10, materially below question's .358 -- once the gate
#          directions are excluded by the sign split, the payload channel is NOT
#          substantially suppressed late.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'channel_depth_p_results.json'
NR = 960
SITE = 17
RANK = 8          # pos_r8 payload slice (S1598 rule for mlp17 sites)
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


def _stack_order():
    order = ['x0']
    for L in range(18):
        order += [f'attn{L}', f'mlp{L}']
    return order


def _layer_of(name):
    if name == 'x0':
        return -1
    return int(name[4:]) if name.startswith('attn') else int(name[3:])


@torch.no_grad()
def depth_curve(rows, V2, lam2, mask_v):
    """Per-component signed class contribution, summed in stack order."""
    comps = ['x0'] + [f'attn{L}' for L in range(18)] + [f'mlp{L}' for L in range(18)]
    acc = {'sum': {c: torch.zeros(RANK, device=DEV) for c in comps},
           'csum': {c: torch.zeros(RANK, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(RANK, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(RANK, device=DEV) for h in range(9)} for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]
        pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)

    # exact depth-decay coefficients from the block lambdas (same as S1601)
    lam0 = [float(blk.lambdas[0]) for blk in H]
    lam1 = [float(blk.lambdas[1]) for blk in H]
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
    signed = {c: (coef[c] * (cmu[c] - mu[c])).sum().item() for c in comps}

    order = _stack_order()
    steps = [signed[c] for c in order]
    cum, run = [], 0.0
    for s in steps:
        run += s
        cum.append(run)
    peak_i = max(range(len(cum)), key=lambda i: cum[i])
    drop_i = min(range(len(steps)), key=lambda i: steps[i])
    peak, final = cum[peak_i], cum[-1]
    drawdown = (peak - final) / peak if peak > 0 else float('nan')
    return {'order': order, 'steps': steps, 'cum': cum,
            'peak_component': order[peak_i], 'peak_layer': _layer_of(order[peak_i]),
            'peak_value': peak, 'final_value': final, 'drawdown': drawdown,
            'largest_drop_component': order[drop_i],
            'largest_drop_value': steps[drop_i],
            'recon_rel_err': rec_err, 'class_n': acc['cn'], 'signed': signed}



SAMPLES = [80, 15000, 20000, 25000, 30000, 35000]   # disjoint; eval uses 7000-7960


@torch.no_grad()
def main():
    import os, math
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    mask_v = rx(r'^ (he|she|they|He|She|They)$')
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()

    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    wdir = u @ Dw
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    # S1598 RULE: mlp17 slices must be SIGN-SPLIT. Take the RANK most POSITIVE
    # eigendirections (payload), NOT |lambda|-ordered -- the |lambda|-top slice
    # here is suppression-dominated (6 of 8 negative) and mixes gate with payload.
    o2 = lam.argsort(descending=True)[:RANK]
    V2 = V[:, o2].contiguous(); lam2 = lam[o2].contiguous()
    print('payload eigs (pos_r%d)' % RANK,
          [round(float(x_), 4) for x_ in lam2], flush=True)
    assert bool((lam2 > 0).all()), 'payload slice must be all-positive'

    per = []
    for skip in SAMPLES:
        rows = cl.fineweb_rows(96, skip=skip)[:, :T + 1].contiguous()
        r = depth_curve(rows, V2, lam2, mask_v)
        signed = r.pop('signed')
        pos_ranked = sorted([c for c in signed if signed[c] > 0],
                            key=lambda c: -signed[c])
        rec = {'skip': skip, 'class_n': r['class_n'],
               'recon_rel_err': r['recon_rel_err'],
               'peak_component': r['peak_component'], 'peak_layer': r['peak_layer'],
               'peak_value': r['peak_value'], 'final_value': r['final_value'],
               'drawdown': r['drawdown'],
               'largest_drop_component': r['largest_drop_component'],
               'largest_drop_value': r['largest_drop_value'],
               'top2_positive': pos_ranked[:2],
               'mlp17_signed': round(signed.get('mlp17', 0.0), 2),
               'mlp17_pos_rank': (pos_ranked.index('mlp17') + 1
                                  if 'mlp17' in pos_ranked else None)}
        per.append(rec)
        print(f"skip={skip:6d} n={rec['class_n']:3d} peak={rec['peak_component']:7s}"
              f"(L{rec['peak_layer']:2d}) dd={rec['drawdown']:+.4f} "
              f"top2+={pos_ranked[:2]} mlp17={rec['mlp17_signed']:+.1f}"
              f"(rank {rec['mlp17_pos_rank']}) recon={rec['recon_rel_err']:.1e}", flush=True)

    dd = [p['drawdown'] for p in per]
    n = len(dd); mean = sum(dd) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in dd) / (n - 1))
    sem = sd / math.sqrt(n)
    n_late = sum(1 for p in per if p['peak_layer'] >= 14)
    n_m17top2 = sum(1 for p in per if p['mlp17_pos_rank'] in (1, 2))

    pa = n_late >= 5
    pb = n_m17top2 >= 5
    pc = mean <= 0.10

    out = {'config': {'samples': SAMPLES, 'rows_per_sample': 96,
                      'site': SITE, 'rank': RANK, 'slice': 'pos_r8 (all-positive payload)',
                      'class': 'pronouns'},
           'per_sample': per,
           'drawdown': {'values': [round(x, 4) for x in dd], 'mean': round(mean, 4),
                        'sd': round(sd, 4), 'sem': round(sem, 4),
                        'min': round(min(dd), 4), 'max': round(max(dd), 4)},
           'peak_late_count': n_late, 'mlp17_top2_count': n_m17top2,
           'question_reference': {'drawdown_mean': 0.3584, 'peak': 'attn16',
                                  'section': 'S1603'},
           'predictions': {'pred_a_peak_ge_L14_5of6': bool(pa),
                           'pred_b_mlp17_top2_positive_5of6': bool(pb),
                           'pred_c_mean_drawdown_le_10pct': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"drawdown mean {mean:+.4f} sd {sd:.4f} sem {sem:.4f} "
          f"(question ref .3584) | peak>=L14 {n_late}/6 | mlp17 top2+ {n_m17top2}/6", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
