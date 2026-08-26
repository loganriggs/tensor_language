# slice_writers: SPARSE CONNECTIONS BETWEEN LOW-RANK EIGENSPACES (user
# 2026-08-26: "define variables/features by sparse connections between low
# rank eigenspaces... connect multiple mlps and attn that we know connect and
# find a sparse eigenbasis there.") The certified question@mlp11 rank-2 eigen
# slice (S1573) reads coordinates z.v1, z.v2 at the mlp11 input. Those
# coordinates are written by upstream components: the pre-norm input is
#   P = xm_11 + ao_11,  an EXACT linear combination of component outputs
#   {x0, ao_0..ao_11, mo_0..mo_10} with scalar coefficients from the lambdas.
# So the slice defines a candidate GRAPH EDGE SET: which components carry the
# class signal into the slice subspace?
#   1. ATTRIBUTION: per component j, delta_j,i = coef_j * (class-mean minus
#      global-mean of o_j.v_i) over fit rows; exact reconstruction check.
#      Head-grain scores for every attn layer (via c_proj weight slices).
#   2. CAUSAL EDGES: mean-substitute the span(v1,v2) component of ONLY the
#      top-4 writer outputs (at their source, so effects propagate); compare
#      the question class CE rise to the full rank-2 form ablation reference
#      (same rows) and to a mid-ranked 4-component control edit.
# NR=960 eval, 96 fit rows, question class.
# Registered predictions:
#   pred_a SPARSITY: the top-4 of ~24 components carry >= .60 of the total
#          attribution mass sum_j (|delta_j,1| + |delta_j,2|).
#   pred_b CAUSAL: editing span(v) out of the top-4 writers reproduces
#          >= .50 of the form_r2 ablation class rise, selectivity >= 20x.
#   pred_c ATTENTION EDGE: at least one attn layer is in the causal top-4,
#          and its top head at head grain is one of the 5 certified question
#          circuit heads (compression_rank2 W list).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'slice_writers_results.json'
NR = 960
SITE = 11
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
EDIT = {'set': set(), 'V': None, 'mu': None}   # mu: {name: [2]}
FORM = {'on': False, 'V': None, 'lam': None, 'mean_s': 0.0, 'u': None}


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
        if L == SITE and FORM['on']:
            z = args[0].float()
            zv = z @ FORM['V']
            s = (zv * zv) @ FORM['lam']
            o = (output.float() if o is None else o) \
                - (s - FORM['mean_s']).unsqueeze(-1) * FORM['u']
        return None if o is None else o.to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
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
    P = None
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
        if L == SITE:
            P = x
            z = F.rms_norm(x, (D,)).float()
            zv = z.reshape(-1, D) @ V2
            acc['s_sum'] += float((((zv * zv) @ lam2)).sum())
            acc['s_n'] += zv.shape[0]
            break
        mo = blk.mlp(F.rms_norm(x, (D,)))
        add(f'mlp{L}', mo)
        x = x + mo
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

    comps = ['x0'] + [f'attn{L}' for L in range(SITE + 1)] \
        + [f'mlp{L}' for L in range(SITE)]
    acc = {'sum': {c: torch.zeros(2, device=DEV) for c in comps},
           'csum': {c: torch.zeros(2, device=DEV) for c in comps},
           'hsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)}
                    for L in range(SITE + 1)},
           'hcsum': {L: {h: torch.zeros(2, device=DEV) for h in range(9)}
                     for L in range(SITE + 1)},
           's_sum': 0.0, 's_n': 0, 'n': 0, 'cn': 0, 'P_proj': []}
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
    coef = {'attn11': 1.0}
    for l in range(SITE):
        c = 1.0
        for k in range(l + 1, SITE + 1):
            c *= lam0[k]
        coef[f'attn{l}'] = c; coef[f'mlp{l}'] = c
    tx0 = 1.0
    for k in range(SITE):
        tx0 = lam0[k] * tx0 + lam1[k]
    coef['x0'] = lam0[SITE] * tx0 + lam1[SITE]
    # note: mlp_l is added AFTER xm_l, so its coefficient excludes lam0_l:
    # both ao_l and mo_l enter x_{l+1} with weight 1 -> same coef. correct.

    # reconstruction check: sum_j coef_j * sum(o_j.v) vs sum(P.v)
    recon = sum(coef[c] * acc['sum'][c] for c in comps)
    Pv = torch.stack(acc['P_proj']).sum(0)
    rec_err = float((recon - Pv).abs().max() / Pv.abs().max())
    print('reconstruction rel err', round(rec_err, 6), flush=True)

    mu = {c: acc['sum'][c] / max(acc['n'], 1) for c in comps}
    cmu = {c: acc['csum'][c] / max(acc['cn'], 1) for c in comps}
    delta = {c: (coef[c] * (cmu[c] - mu[c])).abs().sum().item() for c in comps}
    ranked = sorted(comps, key=lambda c: -delta[c])
    tot = sum(delta.values())
    top4 = ranked[:4]
    top4_share = sum(delta[c] for c in top4) / max(tot, 1e-9)
    attr = {c: {'delta_abs': round(delta[c], 4),
                'delta_signed': [round(float(x_), 4) for x_ in
                                 (coef[c] * (cmu[c] - mu[c]))]}
            for c in ranked[:10]}
    print('top writers', json.dumps({c: attr[c]['delta_abs']
                                     for c in ranked[:8]}), flush=True)

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
             for L in range(SITE + 1)]
    hooks += [H[L].mlp.register_forward_hook(mk_mlp_hook(L))
              for L in range(SITE + 1)]

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
    FORM.update({'on': True, 'V': V2, 'lam': lam2,
                 'mean_s': acc['s_sum'] / max(acc['s_n'], 1), 'u': u})
    g1, c1 = measure()
    FORM['on'] = False
    ref_rise = c1 - c0
    res = {'slice_eigs': [round(float(x_), 4) for x_ in lam2],
           'recon_rel_err': round(rec_err, 6),
           'attr_top10': attr, 'head_grain': {c: [[h, round(s, 4)]
                                                  for h, s in v]
                                              for c, v in heads.items()},
           'top4': top4, 'top4_share': round(top4_share, 4),
           'form_r2_ref': {'rise_class': round(ref_rise, 4),
                           'rise_global': round(g1 - g0, 4)}}
    print('form ref', res['form_r2_ref'], flush=True)

    EDIT['V'] = V2
    EDIT['mu'] = {c: mu[c] for c in comps}
    edit_top = [c for c in ranked if c != 'x0'][:4]
    for nm_, eset in (('edit_top4', edit_top),
                      ('edit_mid4', [c for c in ranked if c != 'x0'][8:12])):
        EDIT['set'] = set(eset)
        g2, c2 = measure()
        EDIT['set'] = set()
        res[nm_] = {'set': eset, 'rise_class': round(c2 - c0, 4),
                    'rise_global': round(g2 - g0, 4),
                    'selectivity': round((c2 - c0) / max(g2 - g0, 1e-6), 2),
                    'frac_of_form': round((c2 - c0) / max(ref_rise, 1e-6), 3)}
        print(nm_, res[nm_], flush=True)
    for hk in hooks:
        hk.remove()

    r2 = json.load(open(PT + 'compression_rank2_results.json'))['res']
    circ = set(r2['question']['W']['heads'])
    pa = top4_share >= 0.60
    pb = res['edit_top4']['frac_of_form'] >= 0.50 and \
        res['edit_top4']['selectivity'] >= 20
    attn_in = [c for c in edit_top if c.startswith('attn')]
    pc = bool(attn_in) and any(
        f"{c[4:]}.{heads.get(c, [[None]])[0][0]}" in circ for c in attn_in)
    out = {'res': res, 'circuit_heads_ref': sorted(circ),
           'pred_a_sparse_60': bool(pa), 'pred_b_causal_edges': bool(pb),
           'pred_c_attn_edge_is_circuit_head': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
