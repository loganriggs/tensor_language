# headgrain_control: DOES THE HEAD-GRAIN LAW SURVIVE A RANDOM-SUBSPACE CONTROL?
# S1606 showed a RANDOM rank-8 subspace returns mlp17/mlp16/mlp15 as its top
# "writers" -- component attribution has a generic high-norm floor, and once it is
# removed two real slice rules share ZERO components. LESSONS 15 now requires a
# random control for every component-level writer claim.
#
# The head-grain law is the strongest claim in this arc and has never had that
# control. S1597: attn10 -> head 10.5 at 20:1 over the next head; attn9 -> 9.7.
# S1598: attn9 -> 9.6 at 6.8:1; attn12 -> 12.4 at 9.1:1, and "on BOTH classes
# tested, every attention writer of a certified eigen slice resolves at head grain
# to a certified circuit head of that class."
#
# LESSONS 15 argues head grain should be LESS exposed than component attribution,
# because it is a WITHIN-layer ratio in which a global norm floor largely cancels
# -- but that is a hypothesis, not a result. This tests it.
#
# Cell: pronouns @ mlp17, pos_r8 payload vs a random orthonormal rank-8 basis
# (same seed family as S1606), 3 disjoint 96-row samples each. Head grain is the
# signed class-minus-global head contribution, |.|-summed over slice directions,
# exactly as S1597/S1598 computed it.
# Certified pronoun heads (S1598 roster): 6.7, 7.3, 9.6, 12.4, 13.2.
#
# Registered predictions:
#   pred_a the RANDOM basis's median top-head:second-head ratio across the 18
#          attention layers is < 3.0 -- materially below the 6.8-20:1 the real
#          slices report. (If random also reaches 6-20:1, the law is an artifact.)
#   pred_b at attn9 and attn12 the payload slice's top head is a certified
#          pronoun head in 3/3 samples, while the random basis's top head at
#          those layers is certified in <= 1 of 3.
#   pred_c the payload slice's top-head ratio exceeds the random basis's at the
#          SAME layer for >= 12 of 18 layers.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'headgrain_control_results.json'
NR = 960
SITE = 17
RANK = 8
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
    # head grain, computed exactly as S1597/S1598: signed class-minus-global head
    # contribution, |.|-summed over the slice directions, times the depth coefficient
    hg = {}
    for L in range(18):
        hd = {h: float((coef[f'attn{L}'] * (acc['hcsum'][L][h] / max(acc['cn'], 1)
                        - acc['hsum'][L][h] / max(acc['n'], 1))).abs().sum())
              for h in range(9)}
        rk = sorted(hd.items(), key=lambda kv: -kv[1])
        hg[L] = {'top_head': f"{L}.{rk[0][0]}", 'top_val': rk[0][1],
                 'second_val': rk[1][1],
                 'ratio': rk[0][1] / max(rk[1][1], 1e-12)}

    return {'head_grain': hg,
            'order': order, 'steps': steps, 'cum': cum,
            'peak_component': order[peak_i], 'peak_layer': _layer_of(order[peak_i]),
            'peak_value': peak, 'final_value': final, 'drawdown': drawdown,
            'largest_drop_component': order[drop_i],
            'largest_drop_value': steps[drop_i],
            'recon_rel_err': rec_err, 'class_n': acc['cn'], 'signed': signed}



SAMPLES = [15000, 20000, 25000]
PRON = r'^ (he|she|they|He|She|They)$'
CERTIFIED = {'6.7', '7.3', '9.6', '12.4', '13.2'}      # S1598 pronoun roster
FOCUS = [9, 12]                                        # layers S1598 reported


@torch.no_grad()
def main():
    import os, statistics
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    mask_v = rx(PRON)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    o = lam.argsort(descending=True)[:RANK]
    pos_V, pos_lam = V[:, o].contiguous(), lam[o].contiguous()
    assert bool((pos_lam > 0).all()), 'payload must be all-positive'
    gen = torch.Generator(device=DEV).manual_seed(1729)
    rand_V, _ = torch.linalg.qr(torch.randn(D, RANK, device=DEV, generator=gen))
    arms = {'payload': (pos_V, pos_lam),
            'random': (rand_V.contiguous(), torch.ones(RANK, device=DEV))}

    rows_cache = {s: cl.fineweb_rows(96, skip=s)[:, :T + 1].contiguous() for s in SAMPLES}
    res = {}
    for name, (V2, lam2) in arms.items():
        per = []
        for skip in SAMPLES:
            r = depth_curve(rows_cache[skip], V2, lam2, mask_v)
            hg = r['head_grain']
            per.append({L: {'top_head': hg[L]['top_head'],
                            'ratio': round(hg[L]['ratio'], 3)} for L in range(18)})
            focus = {L: (hg[L]['top_head'], round(hg[L]['ratio'], 2)) for L in FOCUS}
            med = statistics.median([hg[L]['ratio'] for L in range(18)])
            print(f"{name:8s} skip={skip:6d} n={r['class_n']:3d} median_ratio={med:6.2f} "
                  f"focus={focus}", flush=True)
        res[name] = per
    med_by_arm = {}
    for name in arms:
        allr = [res[name][i][L]['ratio'] for i in range(len(SAMPLES)) for L in range(18)]
        med_by_arm[name] = round(statistics.median(allr), 3)

    rand_med = med_by_arm['random']
    pa = rand_med < 3.0
    pay_cert = sum(1 for i in range(len(SAMPLES))
                   if all(res['payload'][i][L]['top_head'] in CERTIFIED for L in FOCUS))
    rnd_cert = sum(1 for i in range(len(SAMPLES))
                   if all(res['random'][i][L]['top_head'] in CERTIFIED for L in FOCUS))
    pb = pay_cert == 3 and rnd_cert <= 1
    wins = sum(1 for L in range(18)
               if statistics.median([res['payload'][i][L]['ratio'] for i in range(3)])
               > statistics.median([res['random'][i][L]['ratio'] for i in range(3)]))
    pc = wins >= 12

    out = {'config': {'class': 'pronouns', 'site': SITE, 'rank': RANK,
                      'samples': SAMPLES, 'focus_layers': FOCUS,
                      'certified_pronoun_heads': sorted(CERTIFIED)},
           'per_sample': res, 'median_ratio_by_arm': med_by_arm,
           'focus_certified_samples': {'payload': pay_cert, 'random': rnd_cert},
           'payload_beats_random_layers': wins,
           'reference': {'S1597_attn10': '10.5 at 20:1', 'S1598_attn9': '9.6 at 6.8:1',
                         'S1598_attn12': '12.4 at 9.1:1'},
           'predictions': {'pred_a_random_median_lt3': bool(pa),
                           'pred_b_payload_certified_3of3_random_le1': bool(pb),
                           'pred_c_payload_wins_ge12_layers': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmedian top:second ratio -> {json.dumps(med_by_arm)}", flush=True)
    print(f"focus layers certified in: payload {pay_cert}/3, random {rnd_cert}/3", flush=True)
    print(f"payload beats random at {wins}/18 layers", flush=True)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
