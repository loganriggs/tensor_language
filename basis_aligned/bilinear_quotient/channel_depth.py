# channel_depth: DOES THE '?' CHANNEL'S +/- CANCELLATION ACCUMULATE OR REVERSE?
# S1601 budgeted the final residual's span(v1,v2) coords over all 37 components
# and found two large opposing flows: mlp11 +1028 (biggest positive) against
# mlp17 -1151 (biggest of ANY sign). The top-4 positive cut reproduced only
# .414 of the final cut, so the channel is NOT assembled by a small writer set.
# Open question left in S1601: is the cancellation a monotone accumulate-then-
# suppress profile, or does the coordinate reverse late?
#
# This resolves it with zero new forward machinery: the same fit pass gives the
# per-component signed contribution, and summing it in STACK ORDER gives the
# channel coordinate as a function of depth. Replicated on a disjoint row set.
# NR=960 eval, 96 fit rows (skip=80), held-out 96 rows (skip=15000), question class.
#
# Registered predictions:
#   pred_a the cumulative class-channel curve PEAKS at or before layer 13
#          (mid-stack accumulation, late suppression), and the final value is
#          strictly below the peak on BOTH row sets.
#   pred_b the largest single-component DROP in the cumulative curve is at
#          mlp17 (its -1151 attribution is the dominant suppressive step).
#   pred_c the peak-to-final drawdown (peak - final)/peak is >= .25, and the
#          held-out drawdown agrees within 20% relative.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'channel_depth_results.json'
NR = 960
SITE = 11
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
    acc = {'sum': {c: torch.zeros(2, device=DEV) for c in comps},
           'csum': {c: torch.zeros(2, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)} for L in range(18)},
           'hcsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)} for L in range(18)},
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    mask_v = rx(r'^\?$| \?$')
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()

    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    wdir = u @ Dw
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    order2 = lam.abs().argsort(descending=True)[:2]
    V2 = V[:, order2].contiguous(); lam2 = lam[order2].contiguous()
    print('slice eigs', [round(float(x_), 4) for x_ in lam2], flush=True)

    out = {}
    for tag, skip in (('fit', 80), ('heldout', 15000)):
        rows = cl.fineweb_rows(96, skip=skip)[:, :T + 1].contiguous()
        r = depth_curve(rows, V2, lam2, mask_v)
        out[tag] = r
        print(f"[{tag}] class_n={r['class_n']} recon_rel_err={r['recon_rel_err']:.2e}", flush=True)
        print(f"[{tag}] peak {r['peak_component']} (layer {r['peak_layer']}) "
              f"= {r['peak_value']:.1f} -> final {r['final_value']:.1f} "
              f"| drawdown {r['drawdown']:.3f}", flush=True)
        print(f"[{tag}] largest single drop: {r['largest_drop_component']} "
              f"({r['largest_drop_value']:.1f})", flush=True)
        curve = {c: round(v, 1) for c, v in zip(r['order'], r['cum'])
                 if c in ('x0', 'mlp5', 'mlp9', 'mlp11', 'mlp13', 'mlp15', 'mlp16', 'mlp17')}
        print(f"[{tag}] cum curve @ landmarks {json.dumps(curve)}", flush=True)

    f, h = out['fit'], out['heldout']
    pa = (f['peak_layer'] <= 13 and h['peak_layer'] <= 13
          and f['final_value'] < f['peak_value'] and h['final_value'] < h['peak_value'])
    pb = f['largest_drop_component'] == 'mlp17' and h['largest_drop_component'] == 'mlp17'
    rel = abs(h['drawdown'] - f['drawdown']) / max(abs(f['drawdown']), 1e-9)
    pc = f['drawdown'] >= 0.25 and rel <= 0.20

    for k in ('fit', 'heldout'):
        out[k].pop('signed', None)
    out['predictions'] = {'pred_a_peak_by_L13_and_falls': bool(pa),
                          'pred_b_largest_drop_is_mlp17': bool(pb),
                          'pred_c_drawdown_ge_25pct_replicates': bool(pc),
                          'drawdown_rel_gap': round(rel, 4)}
    out['runtime_s'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc} (rel gap {rel:.3f})", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)


if __name__ == '__main__':
    main()
