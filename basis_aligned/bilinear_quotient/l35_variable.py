"""FAN-OUT C (dossier mlp-transition-L3-5): complete the transition account. §1095: L4 consumes the content
PRECURSOR in its local rotated coordinates (own-64 recovery 0.74; precursor overlap 0.645). Is the WHOLE
transition band one shared rotating variable? For L3 and L5 (mirroring l4_variable): (1) own-basis rank profile
(own-16/64/256 dev recovery of each layer's mean-ablation gap); (2) adjacent-coordinate identity — overlap of
each layer's own top-64 dev basis with its neighbors' (L3<->L4, L4<->L5), with the deep ref (L8-12) and grammar
(L0-1); (3) causal cross-projection: substitute L3's dev projected on L4's basis (and L5's on L4's) -> does a
neighbor's basis carry the function? NSEQ=192.

REGISTERED PREDICTIONS:
  (0) SANITY: full ~0 cost; own-256 >= own-64 >= own-16; random-64 lowest.
  (a) ONE ROTATING VARIABLE: adjacent own-basis overlaps (L3-L4, L4-L5) >= 0.55 (vs random 0.056) and each
      neighbor-basis causal projection recovers >= 0.8x the layer's own-64 recovery -> the transition band reads
      and writes ONE shared content precursor that rotates smoothly (extends §1049's deep-band sharing down
      through the transition; §1052's drift fully functional);
  (b) if L3's own basis overlaps GRAMMAR (L0-1) more than L4's basis, L3 is still a grammar-side layer and the
      variable handoff happens between L3 and L4 (report the boundary);
  (c) if own-64 recovery < 0.5 for L3/L5, those layers are broadband (unlike L4's 0.74; report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l35_variable_results.json'
NSEQ = 192; SEQ = 256; K = 64
TARGETS = [3, 5]; ALLBASIS = [0, 1, 3, 4, 5, 8, 10, 12]
H = m.transformer.h
SUB = {'layer': -1, 'U': None, 'mode': None}
ST = {'xbar': {}, 'obar': {}}
CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        mt = ST['xbar'][L][CUR['tok']].to(x.dtype)
        if SUB['mode'] == 'meanabl':
            return ST['obar'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        if SUB['mode'] == 'full': xin = x
        elif SUB['mode'] == 'mtok': xin = mt
        else:
            U = SUB['U']; dv = (x - mt).float()
            xin = mt + ((dv @ U) @ U.T).to(x.dtype)
        y = mo.Down(mo.Left(xin)*mo.Right(xin)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


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

    cap = {L: [] for L in ALLBASIS}; capO = {L: [] for L in TARGETS}; hs = []
    for L in ALLBASIS:
        def mk(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                if L in TARGETS: capO[L].append(o_.detach().float().reshape(-1, D))
                return None
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    bases = {}; S_own = {}
    def dev_of(layers):
        devsum = None
        for L in layers:
            X = torch.cat(cap[L], 0)
            xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
            xb = xb/cn.clamp_min(1).unsqueeze(1)
            if len(layers) == 1 and L in TARGETS + [4]:
                ST['xbar'][L] = xb.half()
            dv = X - xb[tok]
            devsum = dv if devsum is None else devsum + dv
        dev = devsum/len(layers); return dev - dev.mean(0)
    for L in [3, 4, 5]:
        dev = dev_of([L]); _, S, Vt = torch.linalg.svd(dev, full_matrices=False)
        bases[f'own{L}'] = Vt; S_own[L] = S; del dev
    devD = dev_of([8, 10, 12]); _, _, VtD = torch.linalg.svd(devD, full_matrices=False); bases['deep'] = VtD; del devD
    devG = dev_of([0, 1]); _, _, VtG = torch.linalg.svd(devG, full_matrices=False); bases['gram'] = VtG; del devG
    g = torch.Generator(device=DEV).manual_seed(0)
    Ur = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    for L in TARGETS: ST['obar'][L] = torch.cat(capO[L], 0).mean(0)
    for L in ALLBASIS: cap[L] = []

    def ov(A, B): return round(float((A.T @ B).pow(2).sum()/K), 4)
    U = {n: bases[n][:K].T.contiguous() for n in bases}
    overlaps = {'L3_L4': ov(U['own3'], U['own4']), 'L4_L5': ov(U['own4'], U['own5']),
                'L3_deep': ov(U['own3'], bases['deep'][:K].T.contiguous()), 'L5_deep': ov(U['own5'], bases['deep'][:K].T.contiguous()),
                'L3_gram': ov(U['own3'], bases['gram'][:K].T.contiguous()), 'L4_gram': ov(U['own4'], bases['gram'][:K].T.contiguous()),
                'L5_gram': ov(U['own5'], bases['gram'][:K].T.contiguous()), 'L3_rand': ov(U['own3'], Ur)}

    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in TARGETS]
    SUB['layer'] = -1; base = ce(blocks)
    res = {}
    for L in TARGETS:
        row = {}
        conds = {'full': None, 'mtok': None, 'own16': bases[f'own{L}'][:16].T.contiguous(),
                 'own64': U[f'own{L}'], 'own256': bases[f'own{L}'][:256].T.contiguous(),
                 'neighbor64': U['own4'], 'deep64': bases['deep'][:K].T.contiguous(),
                 'gram64': bases['gram'][:K].T.contiguous(), 'rand64': Ur, 'meanabl': None}
        for mode, UU in conds.items():
            SUB['layer'] = L; SUB['mode'] = mode; SUB['U'] = UU
            row[mode] = round(ce(blocks) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        abl = max(row['meanabl'], 1e-6)
        row_recov = {mode: round(1 - row[mode]/abl, 3) for mode in conds if mode != 'meanabl'}
        res[str(L)] = {'costs': row, 'recov': row_recov}
        print(f"L{L} recov: {row_recov} (meanabl {row['meanabl']})", flush=True)
    for h in hs: h.remove()

    out = {'base_ce': round(base, 4), 'per_layer': res, 'overlaps': overlaps,
           'own64_varfrac': {str(L): round(float((S_own[L][:64]**2).sum()/(S_own[L]**2).sum()), 4) for L in [3, 4, 5]}}
    r3 = res['3']['recov']; r5 = res['5']['recov']
    out['pred_a_one_rotating_variable'] = bool(overlaps['L3_L4'] >= 0.55 and overlaps['L4_L5'] >= 0.55
                                               and r3['neighbor64'] >= 0.8*r3['own64'] and r5['neighbor64'] >= 0.8*r5['own64'])
    out['pred_b_L3_grammar_side'] = bool(overlaps['L3_gram'] > overlaps['L4_gram']*1.5)
    out['pred_c_broadband'] = bool(r3['own64'] < 0.5 or r5['own64'] < 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"overlaps: {overlaps}", flush=True)
    print(f"pred_a one-variable {out['pred_a_one_rotating_variable']} | pred_b L3-grammar-side {out['pred_b_L3_grammar_side']} | pred_c broadband {out['pred_c_broadband']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
