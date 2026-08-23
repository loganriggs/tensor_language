"""REDTEAM of §1084 (registered there): the TOK-term CE recoveries used IN-SAMPLE per-token means — for singleton
tokens (~20% of positions) xbar[tok] equals the one observed input, so the TOK term memorizes the sample and its
recovery is inflated. Here: compute XBAR on half A of the data, evaluate the CE substitutions on half B (tokens
unseen in A fall back to the global mean; fraction reported). Layers: 1 (front, 98% claim), 3 (front-like claim),
8, 10 (deep tok-only ~0.59 claims -- the suspect ones), 16 (readout 91%-token claim).

REGISTERED PREDICTIONS:
  (0) SANITY: full reconstruction still ~0 cost (uses actual input, unaffected by xbar source).
  (a) FRONT HOLDS: L1/L3 tok-only recovery stays >= 0.8 with held-out means (front MLPs really are ~static
      per-token functions; §1045 got ~0.9 independently);
  (b) DEEP DROPS: L8/L10 tok-only recovery falls materially (>= 0.15 absolute) with held-out means -> §1084's
      deep tok-only ~0.59 was partly singleton leakage; report corrected numbers;
  (c) READOUT: L16 tok-only recovery stays >= 0.6 (the 91% token-variance is dominated by frequent tokens,
      not singletons). If it also drops sharply, correct §1084's readout claim plainly."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'transition_terms_heldout_results.json'
NSEQ = 192; SEQ = 256  # half A = 96 for means, half B = 96 for eval
LAYERS = [1, 3, 8, 10, 16]
H = m.transformer.h
SUB = {'mode': None, 'layer': -1}
XBAR = {}; XBAR_OUT = {}; GLOB = {}; SEEN = {}; CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def terms(mlp, x, mtok):
    dv = x - mtok
    Lm = mlp.Left(mtok); Rm = mlp.Right(mtok); Ld = mlp.Left(dv); Rd = mlp.Right(dv)
    return mlp.Down(Lm*Rm), mlp.Down(Lm*Rd + Ld*Rm), mlp.Down(Ld*Rd)


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        tokk = CUR['tok']
        mt = XBAR[L][tokk].to(x.dtype)
        unseen = ~SEEN['mask'][tokk]
        if unseen.any(): mt[unseen] = GLOB[L].to(x.dtype)
        if SUB['mode'] == 'meanabl':
            return XBAR_OUT[L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        t_tok, t_cross, t_dev = terms(mo, x, mt)
        b = mo.Down_bias
        if SUB['mode'] == 'tok': y = t_tok + b
        elif SUB['mode'] == 'tokcross': y = t_tok + t_cross + b
        else: y = t_tok + t_cross + t_dev + b
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
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]
    V = int(m.lm_head.weight.shape[0])

    # means from half A only
    cap = {L: [] for L in LAYERS}; capO = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, A.shape[0], 8):
        idx = A[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    seen_counts = torch.zeros(V, device=DEV); seen_counts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    SEEN['mask'] = seen_counts > 0
    for L in LAYERS:
        X = torch.cat(cap[L], 0); cap[L] = []
        xb = torch.zeros(V, D, device=DEV)
        xb.index_add_(0, tok, X)
        XBAR[L] = (xb / seen_counts.clamp_min(1).unsqueeze(1)).half()
        GLOB[L] = X.mean(0)
        XBAR_OUT[L] = torch.cat(capO[L], 0).mean(0); capO[L] = []
        del X

    # eval on half B
    btok = B[:, :-1].to(DEV).reshape(-1)
    unseen_frac = float((~SEEN['mask'][btok]).float().mean())
    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['layer'] = -1; base = ce(B)
    res = {}
    for L in LAYERS:
        row = {}
        for mode in ['full', 'tok', 'tokcross', 'meanabl']:
            SUB['layer'] = L; SUB['mode'] = mode
            row[mode] = round(ce(B) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        abl = max(row['meanabl'], 1e-6)
        row['tok_recov'] = round(1 - row['tok']/abl, 3); row['tokcross_recov'] = round(1 - row['tokcross']/abl, 3)
        res[str(L)] = row
        print(f"L{L} held-out: full {row['full']} | tok {row['tok']} (recov {row['tok_recov']}) | tok+cross {row['tokcross']} (recov {row['tokcross_recov']}) | mean-abl {row['meanabl']}", flush=True)
    for h in hs: h.remove()
    insample = {'1': 0.981, '3': 0.858, '8': 0.586, '10': 0.586, '16': 0.755}  # §1084
    out = {'base_ce': round(base, 4), 'unseen_token_frac': round(unseen_frac, 4), 'ce': res,
           'insample_tok_recov': insample,
           'drop': {L: round(insample[L] - res[L]['tok_recov'], 3) for L in insample}}
    out['pred_a_front_holds'] = bool(res['1']['tok_recov'] >= 0.8 and res['3']['tok_recov'] >= 0.8)
    out['pred_b_deep_drops'] = bool(out['drop']['8'] >= 0.15 and out['drop']['10'] >= 0.15)
    out['pred_c_readout_holds'] = bool(res['16']['tok_recov'] >= 0.6)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"unseen-token frac on eval half: {unseen_frac:.4f}", flush=True)
    print(f"drops vs in-sample: {out['drop']} | pred_a front {out['pred_a_front_holds']} | pred_b deep-drop {out['pred_b_deep_drops']} | pred_c readout {out['pred_c_readout_holds']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
