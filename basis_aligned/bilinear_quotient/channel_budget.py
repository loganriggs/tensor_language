# channel_budget: WHO WRITES THE '?' CHANNEL, OVER THE FULL DEPTH? S1600
# certified span(v1,v2) as the '?' unembed channel (logit-lens rank 1) and
# showed the final-residual cut (+2.02 class) is 2.5x the 4-early-writer
# source cut — later components keep writing the channel. Budget it: exact
# linear decomposition of the FINAL residual's channel coords across all 37
# components (x0, ao_0..17, mo_0..17), signed class attribution per
# component, then a causal cut of the top-4 POSITIVE writers vs the S1600
# final-cut reference.
# NR=960 eval, 96 fit rows, question class.
# Registered predictions:
#   pred_a mlp11 is among the top-2 positive writers of the final channel
#          (its quadratic amplification is the main mid-depth source).
#   pred_b mlp17's signed contribution is NEGATIVE (the gate writes against
#          the channel), and it is the most negative of all 37 components.
#   pred_c cutting span(v) at the top-4 positive writers costs >= .60 x the
#          final-cut class rise, with global rise <= .02.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'channel_budget_results.json'
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    mask_v = rx(r'^\?$| \?$')
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()

    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    wdir = u @ Dw
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    order = lam.abs().argsort(descending=True)[:2]
    V2 = V[:, order].contiguous(); lam2 = lam[order].contiguous()
    print('slice eigs', [round(float(x_), 4) for x_ in lam2], flush=True)

    comps = ['x0'] + [f'attn{L}' for L in range(18)] \
        + [f'mlp{L}' for L in range(18)]
    acc = {'sum': {c: torch.zeros(2, device=DEV) for c in comps},
           'csum': {c: torch.zeros(2, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)}
                    for L in range(18)},
           'hcsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)}
                     for L in range(18)},
           'n': 0, 'cn': 0, 'P_proj': []}
    for i in range(0, 96, 8):
        bb = FR[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        pm = mask_v.to(DEV)[tg]
        pm[:, :64] = False
        capture_fwd(idx, V2, lam2, acc, pm)
    print(f"fit pass done, class n={acc['cn']}", flush=True)

    # exact coefficients from lambdas
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

    # reconstruction check: sum_j coef_j * sum(o_j.v) vs sum(P.v)
    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())
    print('reconstruction rel err', round(rec_err, 6), flush=True)

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    signed = {c: (coef[c] * (cmu[c] - mu[c])).sum().item() for c in comps}
    ranked = sorted(comps, key=lambda c: -signed[c])
    pos_tot = sum(v for v in signed.values() if v > 0)
    top4 = ranked[:4]
    top4_share = sum(max(signed[c], 0.0) for c in top4) / max(pos_tot, 1e-9)
    attr = {c: {'signed': round(signed[c], 4),
                'by_dir': [round(float(x_), 4) for x_ in
                           (coef[c] * (cmu[c] - mu[c]))]}
            for c in ranked[:8] + ranked[-4:]}
    print('top +writers', json.dumps({c: attr[c]['signed']
                                      for c in ranked[:8]}), flush=True)
    print('most negative', json.dumps({c: round(signed[c], 3)
                                       for c in ranked[-4:]}), flush=True)

    # head grain for attn layers in top-8
    heads = {}
    for c in ranked[:8]:
        if c.startswith('attn'):
            L = int(c[4:])
            hd = {h: float((coef[c] * (acc['hcsum'][L][h] / max(acc['cn'], 1)
                            - acc['hsum'][L][h] / max(acc['n'], 1))
                            ).abs().sum()) for h in range(9)}
            heads[c] = sorted(hd.items(), key=lambda kv: -kv[1])[:3]
    print('head grain', json.dumps({c: [[h, round(s, 4)] for h, s in v]
                                    for c, v in heads.items()}), flush=True)

    # causal: hooks
    hooks = [H[L].attn.c_proj.register_forward_hook(mk_cproj_hook(L))
             for L in range(18)]
    hooks += [H[L].mlp.register_forward_hook(mk_mlp_hook(L))
              for L in range(18)]

    def measure():
        gs = 0.0; gn = 0; cs = 0.0; cn_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn_, 1)

    g0, c0 = measure()
    FIN.update({'on': True, 'V': V2, 'mu': Pv / max(acc['n'], 1)})
    g1, c1 = measure()
    FIN['on'] = False
    ref_rise = c1 - c0
    res = {'slice_eigs': [round(float(x_), 4) for x_ in lam2],
           'recon_rel_err': round(rec_err, 6),
           'attr_signed': attr, 'head_grain': {c: [[h, round(s, 4)]
                                                  for h, s in v]
                                              for c, v in heads.items()},
           'top4_positive': top4, 'top4_pos_share': round(top4_share, 4),
           'final_cut_ref': {'rise_class': round(ref_rise, 4),
                             'rise_global': round(g1 - g0, 4)}}
    print('final cut ref', res['final_cut_ref'], flush=True)

    EDIT['V'] = V2
    EDIT['mu'] = {c: mu[c] for c in comps}
    edit_top = [c for c in ranked if c != 'x0' and signed[c] > 0][:4]
    for nm_, eset in (('edit_top4', edit_top),
                      ('edit_neg4',
                       [c for c in reversed(ranked) if c != 'x0'][:4])):
        EDIT['set'] = set(eset)
        g2, c2 = measure()
        EDIT['set'] = set()
        res[nm_] = {'set': eset, 'rise_class': round(c2 - c0, 4),
                    'rise_global': round(g2 - g0, 4),
                    'selectivity': round((c2 - c0) / max(g2 - g0, 1e-6), 2),
                    'frac_of_final': round((c2 - c0) / max(ref_rise, 1e-6), 3)}
        print(nm_, res[nm_], flush=True)
    for hk in hooks:
        hk.remove()

    pa = 'mlp11' in ranked[:2]
    pb = ranked[-1] == 'mlp17' and signed['mlp17'] < 0
    pc = res['edit_top4']['frac_of_final'] >= 0.60 and \
        res['edit_top4']['rise_global'] <= 0.02
    out = {'res': res,
           'pred_a_mlp11_top2': bool(pa),
           'pred_b_mlp17_most_negative': bool(pb),
           'pred_c_top4_cut_60pct': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
