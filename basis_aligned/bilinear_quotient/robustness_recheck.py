"""FAN-OUT D (cross-cutting; user directive: enough data that results won't be overturned). SPLIT-HALF
STABILITY AUDIT of this wave's headline numbers, at 2-4x the original N (384 seqs total, two disjoint halves of
192; originals used 64-96). Re-measured per half, fully independently (bases/means/kernels refit per half):
  (1) L5H7: zero cost (~0.91) and global-mean const-replacement cost (~0.013) [§1089/§1091];
  (2) ALL-CONST static attention cost (~3.67) and all-zero (~3.42), incl. the §1093 inversion;
  (3) L4: own-64 recovery (~0.74), precursor-64 (~0.66) vs deep-64 (~0.36) [§1095];
  (4) L10 middle-pool positional correlation (log-dist r ~ -0.59) vs content-sim (~0.10) [§1085].

REGISTERED PREDICTIONS:
  (0) SANITY: half-A and half-B each reproduce the original numbers' SIGNS and orderings.
  (a) STABLE: every headline number agrees across halves within 20% relative (or 0.05 absolute for
      correlations/recoveries) AND with the original run -> the wave's conclusions are data-stable;
  (b) any number that FAILS the 20% band is flagged UNSTABLE and its ledger claim gets a stated correction
      (report which)."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'robustness_recheck_results.json'
NSEQ = 384; SEQ = 256; K = 64
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CTL = {'mode': None}     # attn: None|'all_const'|'all_zero'|('head57', 'zero'|'const')
MEANS = {}
SUBL4 = {'mode': None, 'U': None}
ST = {}
CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def attn_hook(L):
    def h(mo, args):
        md = CTL['mode']
        if md is None: return None
        y = args[0].clone()
        if md == 'all_const':
            for hh in range(NH): y[..., hh*HD:(hh+1)*HD] = MEANS[L][hh].view(1, 1, HD).to(y.dtype)
        elif md == 'all_zero':
            y[...] = 0.0
        elif isinstance(md, tuple) and md[0] == 'head57' and L == 5:
            if md[1] == 'zero': y[..., 7*HD:8*HD] = 0.0
            else: y[..., 7*HD:8*HD] = MEANS[5][7].view(1, 1, HD).to(y.dtype)
        else:
            return None
        return (y,) + tuple(args[1:])
    return h


def l4_hook(mo, i_, o_):
    if SUBL4['mode'] is None: return None
    x = (i_[0] if isinstance(i_, tuple) else i_)
    mt = ST['xbar4'][CUR['tok']].to(x.dtype)
    if SUBL4['mode'] == 'meanabl':
        return ST['obar4'].view(1, 1, D).expand_as(o_).to(o_.dtype)
    U = SUBL4['U']; dv = (x - mt).float()
    xin = mt + ((dv @ U) @ U.T).to(x.dtype)
    y = mo.Down(mo.Left(xin)*mo.Right(xin)) + mo.Down_bias
    return y.to(o_.dtype)


@torch.no_grad()
def pattern_for(attn, x):
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def audit_half(blocks, tag):
    V = int(m.lm_head.weight.shape[0])
    # capture: head means, mlp inputs at 3/4/5/8/10/12 + L4 outputs, attn input at L10
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in range(18)}
    BASIS_L = [3, 4, 5, 8, 10, 12]
    cap = {L: [] for L in BASIS_L}; capO4 = []; capA10 = []
    hs = []
    for L in range(18):
        def mk(L):
            def h(mo, args): caps[L] += args[0].detach().float().reshape(-1, NH, HD).sum(0)
            return h
        hs.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    for L in BASIS_L:
        def mk2(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                if L == 4: capO4.append(o_.detach().float().reshape(-1, D))
                return None
            return h
        hs.append(H[L].mlp.register_forward_hook(mk2(L)))
    hs.append(H[10].attn.register_forward_pre_hook(lambda mo, args: capA10.append(args[0].detach()) or None))
    idsL = []; npos = 0
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx); npos += idx.numel()
    for h in hs: h.remove()
    for L in range(18): MEANS[L] = caps[L] / npos
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    def basis_of(layers, store_xbar4=False):
        devsum = None
        for L in layers:
            X = torch.cat(cap[L], 0)
            xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
            xb = xb/cn.clamp_min(1).unsqueeze(1)
            if store_xbar4 and L == 4: ST['xbar4'] = xb.half()
            dv = X - xb[tok]
            devsum = dv if devsum is None else devsum + dv
        dev = devsum/len(layers); dev = dev - dev.mean(0)
        _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
        return Vt
    U_own4 = basis_of([4], store_xbar4=True)[:K].T.contiguous()
    U_prec = basis_of([3, 5])[:K].T.contiguous()
    U_deep = basis_of([8, 10, 12])[:K].T.contiguous()
    ST['obar4'] = torch.cat(capO4, 0).mean(0); capO4.clear()

    # (4) L10 positional vs content-sim correlation
    U_c = U_deep
    g = torch.Generator(device=DEV).manual_seed(0)
    feats = []; pats = []
    for x10 in capA10[:8]:
        x = x10.float(); T = x.shape[1]
        pat = pattern_for(H[10].attn, x10).abs().mean(1)
        c = x @ U_c
        for b in range(x.shape[0]):
            qi = torch.randint(6, T, (1500,), generator=g, device=DEV)
            kj = (torch.rand(1500, generator=g, device=DEV) * (qi - 5).float()).long()
            csim = F.cosine_similarity(c[b, qi], c[b, kj], dim=-1)
            feats.append(torch.stack([csim, (qi-kj).float().log1p()], 1).cpu())
            pats.append(pat[b, qi, kj].cpu())
        del pat
    capA10.clear()
    Fm = torch.cat(feats, 0); y = torch.cat(pats, 0)
    Fz = (Fm - Fm.mean(0))/Fm.std(0).clamp_min(1e-6); yz = (y - y.mean())/y.std().clamp_min(1e-6)
    r = (Fz * yz.unsqueeze(1)).mean(0)
    r_content, r_dist = round(float(r[0]), 3), round(float(r[1]), 3)
    for L in BASIS_L: cap[L] = []

    # CE measurements
    hattn = [H[L].attn.c_proj.register_forward_pre_hook(attn_hook(L)) for L in range(18)]
    hl4 = H[4].mlp.register_forward_hook(l4_hook)
    CTL['mode'] = None; base = ce(blocks)
    res = {'base_ce': round(base, 4), 'r_L10_content': r_content, 'r_L10_logdist': r_dist}
    for name, md in [('l5h7_zero', ('head57', 'zero')), ('l5h7_const', ('head57', 'const')),
                     ('all_const', 'all_const'), ('all_zero', 'all_zero')]:
        CTL['mode'] = md
        res[name] = round(ce(blocks) - base, 4)
        CTL['mode'] = None
    for name, U in [('l4_own64', U_own4), ('l4_prec64', U_prec), ('l4_deep64', U_deep), ('l4_meanabl', None)]:
        SUBL4['mode'] = 'meanabl' if U is None else 'proj'; SUBL4['U'] = U
        res[name] = round(ce(blocks) - base, 4)
        SUBL4['mode'] = None
    for h in hattn: h.remove()
    hl4.remove()
    abl = max(res['l4_meanabl'], 1e-6)
    for nm in ['l4_own64', 'l4_prec64', 'l4_deep64']:
        res[nm + '_recov'] = round(1 - res[nm]/abl, 3)
    print(f"[{tag}] {res}", flush=True)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]
    ra = audit_half(A, 'halfA'); rb = audit_half(B, 'halfB')
    originals = {'l5h7_zero': 0.912, 'l5h7_const': 0.013, 'all_const': 3.6725, 'all_zero': 3.421,
                 'l4_own64_recov': 0.74, 'l4_prec64_recov': 0.659, 'l4_deep64_recov': 0.363,
                 'r_L10_content': 0.097, 'r_L10_logdist': -0.586}
    stable = {}; unstable = []
    for k0, orig in originals.items():
        va, vb = ra.get(k0), rb.get(k0)
        if k0.startswith('r_') or k0.endswith('_recov'):
            ok = abs(va-vb) <= 0.07 and abs((va+vb)/2 - orig) <= 0.12
        else:
            mid = (abs(va)+abs(vb))/2
            ok = abs(va-vb) <= 0.2*max(mid, 0.05) and abs(mid-abs(orig)) <= 0.25*max(abs(orig), 0.05)
        stable[k0] = {'halfA': va, 'halfB': vb, 'original': orig, 'stable': bool(ok)}
        if not ok: unstable.append(k0)
    out = {'halfA': ra, 'halfB': rb, 'stability': stable, 'unstable': unstable,
           'inversion_holds': {'halfA': bool(ra['all_zero'] < ra['all_const']), 'halfB': bool(rb['all_zero'] < rb['all_const'])},
           'pred_a_all_stable': bool(len(unstable) == 0),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"unstable: {unstable or 'NONE'} | §1093 inversion holds: {out['inversion_holds']}", flush=True)
    print(f"pred_a all-stable {out['pred_a_all_stable']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
