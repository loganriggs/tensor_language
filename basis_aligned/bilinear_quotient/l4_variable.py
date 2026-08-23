"""Follow-up to §1094 (registered there): L4's context variable is OUTSIDE the deep (L8-12) content basis
(non-content dev recovers 0.91). Which variable is it? Decompose in L4's OWN basis and test identity:
(1) LOW-RANK IN OWN BASIS? substitute mtok + top-K of L4's OWN deviation PCA (K=16/64/256) -> how low-rank is
    the needed context?
(2) IDENTITY: overlap of L4's own top-64 deviation basis with (a) the DEEP content ref basis (L8-12 pooled;
    §1052 measured ~0.3-0.4 for transition layers), (b) the CONTENT PRECURSOR basis (L3's + L5's own deviation
    top-64 — the §1052 drifting early content), (c) the GRAMMAR/class basis (L0-1 MLP-input deviation top-64),
    (d) random null (~0.056).
(3) CAUSAL identity check: substitute mtok + dev projected on the PRECURSOR basis vs on the DEEP basis vs
    on the GRAMMAR basis (all K=64) -> which named basis carries L4's function?

REGISTERED PREDICTIONS:
  (0) SANITY: full ~0; mtok ~0.05 recovery; own-K256 >= own-K64 >= own-K16.
  (a) PRECURSOR READING: L4's needed context is LOW-RANK in its own basis (own-top64 recovery >= 0.7) and its
      basis overlaps the PRECURSOR (L3/L5 own-dev) >= 2x its overlap with the deep L8-12 ref -> L4 consumes the
      rotating EARLY content (the §1052 drift explains §1094; content×content starts at L4 in precursor
      coordinates);
  (b) DIFFERENT-VARIABLE READING: if the grammar-basis projection recovers more than the precursor projection,
      L4 is a grammar/class-context consumer (a genuinely different variable feeding the transition);
  (c) if own-top64 recovery < 0.5, L4's context is high-rank even in its own basis (a broadband consumer;
      report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l4_variable_results.json'
NSEQ = 96; SEQ = 256; L4 = 4; K = 64; RARE_MAX = 2
DEEP = [8, 10, 12]; PREC = [3, 5]; GRAM = [0, 1]
H = m.transformer.h
SUB = {'mode': None}
ST = {}; CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(mo, i_, o_):
    if SUB['mode'] is None: return None
    x = (i_[0] if isinstance(i_, tuple) else i_)
    mt = ST['xbar'][CUR['tok']].to(x.dtype)
    dv = (x - mt).float()
    md = SUB['mode']
    if md == 'meanabl':
        return ST['obar'].view(1, 1, D).expand_as(o_).to(o_.dtype)
    if md == 'full': xin = x
    elif md == 'mtok': xin = mt
    else:
        U = ST[md]
        xin = mt + ((dv @ U) @ U.T).to(x.dtype)
    y = mo.Down(mo.Left(xin)*mo.Right(xin)) + mo.Down_bias
    return y.to(o_.dtype)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    ALL = sorted(set([L4] + DEEP + PREC + GRAM))

    # capture MLP inputs at all needed layers
    cap = {L: [] for L in ALL}; capO = []
    hs = []
    for L in ALL:
        def mk(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                if L == L4: capO.append(o_.detach().float().reshape(-1, D))
                return None
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    def dev_basis(layers, KK=K):
        devsum = None
        for L in layers:
            X = torch.cat(cap[L], 0)
            xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
            dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
            devsum = dv if devsum is None else devsum + dv
        dev = devsum/len(layers); dev = dev - dev.mean(0)
        _, S, Vt = torch.linalg.svd(dev, full_matrices=False)
        return Vt, S

    # L4 own xbar/obar + own bases
    X4 = torch.cat(cap[L4], 0)
    xb4 = torch.zeros(V, D, device=DEV); xb4.index_add_(0, tok, X4)
    ST['xbar'] = (xb4/cn.clamp_min(1).unsqueeze(1)).half()
    ST['obar'] = torch.cat(capO, 0).mean(0)
    Vt4, S4 = dev_basis([L4])
    ST['own16'] = Vt4[:16].T.contiguous(); ST['own64'] = Vt4[:64].T.contiguous(); ST['own256'] = Vt4[:256].T.contiguous()
    VtD, _ = dev_basis(DEEP); ST['deep64'] = VtD[:64].T.contiguous()
    VtP, _ = dev_basis(PREC); ST['prec64'] = VtP[:64].T.contiguous()
    VtG, _ = dev_basis(GRAM); ST['gram64'] = VtG[:64].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    ST['rand64'] = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]

    def ov(A, B): return round(float((A.T @ B).pow(2).sum()/K), 4)
    own = ST['own64']
    overlaps = {'own_vs_deep': ov(own, ST['deep64']), 'own_vs_prec': ov(own, ST['prec64']),
                'own_vs_gram': ov(own, ST['gram64']), 'own_vs_rand': ov(own, ST['rand64'])}
    for L in ALL: cap[L] = []
    del X4

    hk = H[L4].mlp.register_forward_hook(sub_hook)
    SUB['mode'] = None; base = ce(blocks)
    res = {}
    for md in ['full', 'mtok', 'own16', 'own64', 'own256', 'deep64', 'prec64', 'gram64', 'rand64', 'meanabl']:
        SUB['mode'] = md
        res[md] = round(ce(blocks) - base, 4)
        SUB['mode'] = None
        print(f"{md:>8}: cost {res[md]}", flush=True)
    hk.remove()
    abl = max(res['meanabl'], 1e-6)
    recov = {md: round(1 - res[md]/abl, 3) for md in res if md != 'meanabl'}
    out = {'base_ce': round(base, 4), 'costs': res, 'recov': recov, 'overlaps': overlaps,
           'own_dev_top64_varfrac': round(float((S4[:64]**2).sum()/(S4**2).sum()), 4)}
    out['pred_a_precursor'] = bool(recov['own64'] >= 0.7 and overlaps['own_vs_prec'] >= 2*overlaps['own_vs_deep']
                                   and recov['prec64'] > recov['deep64'])
    out['pred_b_grammar'] = bool(recov['gram64'] > recov['prec64'])
    out['pred_c_broadband'] = bool(recov['own64'] < 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recov: {recov}", flush=True)
    print(f"overlaps: {overlaps} | own-top64 varfrac {out['own_dev_top64_varfrac']}", flush=True)
    print(f"pred_a precursor {out['pred_a_precursor']} | pred_b grammar {out['pred_b_grammar']} | pred_c broadband {out['pred_c_broadband']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
