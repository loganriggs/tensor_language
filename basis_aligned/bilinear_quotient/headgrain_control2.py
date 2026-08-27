# headgrain_control2: THE CORRECT CONTROL FOR THE PUBLISHED HEAD-GRAIN LAW.
# S1607 ran a random-subspace control on the PAYLOAD slice and found no head
# concentration beyond random (1.403 vs 1.396) -- but the published law does not
# use the payload slice. S1597 is question@mlp11 rank-2 |lambda|-ordered (eigs
# +144.9/-73.8, head 10.5 at 20:1). S1598 is pronouns@mlp17 rank-8 |lambda|-
# ordered (head 9.6 at 6.8:1, head 12.4 at 9.1:1). S1607 controlled the wrong
# slice; this controls the right one, at BOTH published cells, each against a
# matched-RANK random arm.
#
# 3 disjoint 96-row samples per arm. Head grain computed exactly as S1597/S1598:
# signed class-minus-global head contribution, |.|-summed over slice directions,
# times the depth coefficient.
# Certified: pronouns {6.7,7.3,9.6,12.4,13.2}; question {10.5,12.6,7.6,9.3,9.7}.
#
# Registered predictions:
#   pred_a pronouns@mlp17 |lambda|-r8: attn9's top head is 9.6 AND its top:second
#          ratio is >= 4.0 in >= 2 of 3 samples (reproducing S1598's 6.8:1 within
#          a factor of ~1.7). If the published ratio does not reproduce at all,
#          say so -- that is a bigger finding than the control.
#   pred_b question@mlp11 |lambda|-r2: attn10's top head is 10.5 AND its ratio is
#          >= 10.0 in >= 2 of 3 samples (S1597 reported 20:1).
#   pred_c at BOTH cells the |lambda| arm's median top:second ratio exceeds its
#          matched-rank random arm's by a factor >= 1.5.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'headgrain_control2_results.json'
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
    rk = V2.shape[1]          # infer from the basis; cells differ in rank (2 vs 8)
    acc = {'sum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'csum': {c: torch.zeros(rk, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(rk, device=DEV) for h in range(9)} for L in range(18)},
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
CELLS = {
    'pronouns@mlp17': {'site': 17, 'rank': 8,
                       'pat': r'^ (he|she|they|He|She|They)$',
                       'certified': {'6.7', '7.3', '9.6', '12.4', '13.2'},
                       'focus': [9, 12], 'ref': 'S1598 9.6@6.8:1, 12.4@9.1:1'},
    'question@mlp11': {'site': 11, 'rank': 2,
                       'pat': r'^\?$| \?$',
                       'certified': {'10.5', '12.6', '7.6', '9.3', '9.7'},
                       'focus': [10, 9], 'ref': 'S1597 10.5@20:1'},
}


@torch.no_grad()
def main():
    import os, statistics
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows_cache = {s: cl.fineweb_rows(96, skip=s)[:, :T + 1].contiguous() for s in SAMPLES}
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    out_cells = {}

    for cname, cfg in CELLS.items():
        site, rank = cfg['site'], cfg['rank']
        mask_v = rx(cfg['pat'])
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[site].mlp.Left.weight.float(); Rw = H[site].mlp.Right.weight.float()
        Dw = H[site].mlp.Down.weight.float()
        Q = Lw.T @ ((u @ Dw)[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        o = lam.abs().argsort(descending=True)[:rank]      # THE PUBLISHED RULE
        abs_V, abs_lam = V[:, o].contiguous(), lam[o].contiguous()
        gen = torch.Generator(device=DEV).manual_seed(1729)
        rnd_V, _ = torch.linalg.qr(torch.randn(D, rank, device=DEV, generator=gen))
        print(f"\n=== {cname} (rank {rank}, |lambda| rule) eigs "
              f"{[round(float(x), 2) for x in abs_lam]}  ref: {cfg['ref']}", flush=True)

        arms = {'lambda': (abs_V, abs_lam),
                'random': (rnd_V.contiguous(), torch.ones(rank, device=DEV))}
        cell = {}
        for aname, (V2, lam2) in arms.items():
            per = []
            for skip in SAMPLES:
                r = depth_curve(rows_cache[skip], V2, lam2, mask_v)
                hg = r['head_grain']
                per.append({L: {'top_head': hg[L]['top_head'],
                                'ratio': round(hg[L]['ratio'], 3)} for L in range(18)})
                foc = {L: (hg[L]['top_head'], round(hg[L]['ratio'], 2)) for L in cfg['focus']}
                med = statistics.median([hg[L]['ratio'] for L in range(18)])
                print(f"  {aname:7s} skip={skip:6d} n={r['class_n']:3d} "
                      f"median={med:6.2f} focus={foc}", flush=True)
            allr = [per[i][L]['ratio'] for i in range(len(SAMPLES)) for L in range(18)]
            cell[aname] = {'per_sample': per, 'median_ratio': round(statistics.median(allr), 3)}
        cell['ratio_lambda_over_random'] = round(
            cell['lambda']['median_ratio'] / max(cell['random']['median_ratio'], 1e-9), 3)
        cell['eigs'] = [round(float(x), 3) for x in abs_lam]
        cell['ref'] = cfg['ref']
        print(f"  -> {cname}: median lambda {cell['lambda']['median_ratio']} vs "
              f"random {cell['random']['median_ratio']} "
              f"(x{cell['ratio_lambda_over_random']})", flush=True)
        out_cells[cname] = cell

    P = out_cells['pronouns@mlp17']['lambda']['per_sample']
    pa = sum(1 for i in range(3) if P[i][9]['top_head'] == '9.6' and P[i][9]['ratio'] >= 4.0) >= 2
    Qc = out_cells['question@mlp11']['lambda']['per_sample']
    pb = sum(1 for i in range(3) if Qc[i][10]['top_head'] == '10.5' and Qc[i][10]['ratio'] >= 10.0) >= 2
    pc = all(out_cells[c]['ratio_lambda_over_random'] >= 1.5 for c in CELLS)

    out = {'config': {'samples': SAMPLES, 'rows_per_sample': 96,
                      'rule': '|lambda|-ordered (the published rule)',
                      'cells': {k: {kk: (sorted(vv) if isinstance(vv, set) else vv)
                                    for kk, vv in v.items() if kk != 'pat'}
                                for k, v in CELLS.items()}},
           'cells': out_cells,
           'predictions': {'pred_a_pron_attn9_is_9.6_ratio_ge4_2of3': bool(pa),
                           'pred_b_ques_attn10_is_10.5_ratio_ge10_2of3': bool(pb),
                           'pred_c_lambda_beats_random_1.5x_both_cells': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\npred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
