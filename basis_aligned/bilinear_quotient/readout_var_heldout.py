"""Settles §1088's downgraded claims: §1084 reported L16 (readout MLP) output variance = 91% TOKEN term and
'dev-only substitution 0.874 >> mean-abl 0.152' -- both computed with IN-SAMPLE per-token means (singleton leak,
20% of positions). Here both measurements are redone with HELD-OUT means (xbar from half A, measured on half B;
unseen tokens -> global mean). Layers: 16 (the claim), 8 (deep control), 1 (front control, expected robust).

REGISTERED PREDICTIONS:
  (0) SANITY: full reconstruction ~0; L1 tok variance share stays high (> 0.6) held-out (front is genuinely
      token-driven, §1045/§1088).
  (a) READOUT CLAIM SURVIVES OR DIES: if L16 held-out tok variance share stays > 0.7 AND dev-only substitution
      still costs >> mean-ablation, §1084's readout picture is REINSTATED (the token calibration is real, just
      its CE recovery was leak-inflated); if the tok share collapses (< 0.5), the whole readout-as-token-lookup
      picture was leak and §1088's withdrawal is final;
  (b) DEEP: L8 held-out tok share < in-sample (0.33), dev share rises -> deep variance is context, matching the
      held-out CE result (L8 tok recovery 0.14)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_var_heldout_results.json'
NSEQ = 192; SEQ = 256; LAYERS = [1, 8, 16]
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


def heldout_mt(L, tokk, dtype):
    mt = XBAR[L][tokk].to(dtype)
    unseen = ~SEEN['mask'][tokk]
    if unseen.any(): mt[unseen] = GLOB[L].to(dtype)
    return mt


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        mt = heldout_mt(L, CUR['tok'], x.dtype)
        if SUB['mode'] == 'meanabl':
            return XBAR_OUT[L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        t_tok, t_cross, t_dev = terms(mo, x, mt)
        b = mo.Down_bias
        if SUB['mode'] == 'tok': y = t_tok + b
        elif SUB['mode'] == 'dev': y = t_dev + b
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

    # means from A
    cap = {L: [] for L in LAYERS}; capO = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsA = []
    for i in range(0, A.shape[0], 8):
        idx = A[i:i+8].to(DEV)[:, :-1].contiguous(); idsA.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tokA = torch.cat(idsA, 0)
    sc = torch.zeros(V, device=DEV); sc.index_add_(0, tokA, torch.ones_like(tokA, dtype=torch.float))
    SEEN['mask'] = sc > 0
    for L in LAYERS:
        X = torch.cat(cap[L], 0); cap[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tokA, X)
        XBAR[L] = (xb / sc.clamp_min(1).unsqueeze(1)).half()
        GLOB[L] = X.mean(0)
        XBAR_OUT[L] = torch.cat(capO[L], 0).mean(0); capO[L] = []
        del X

    # variance shares on B with held-out means
    capB = {L: [] for L in LAYERS}; capOB = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                capB[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capOB[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsB = []
    for i in range(0, B.shape[0], 8):
        idx = B[i:i+8].to(DEV)[:, :-1].contiguous(); idsB.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tokB = torch.cat(idsB, 0)
    var_shares = {}
    for L in LAYERS:
        X = torch.cat(capB[L], 0); capB[L] = []; capOB[L] = []
        mlp = H[L].mlp; sums = torch.zeros(3, device=DEV)
        for i in range(0, X.shape[0], 4096):
            xx = X[i:i+4096]; mt = heldout_mt(L, tokB[i:i+4096], torch.float32)
            tt, tc, td = terms(mlp, xx, mt)
            for j, t in enumerate((tt, tc, td)): sums[j] += (t.float()**2).sum()
        s = sums/sums.sum()
        var_shares[str(L)] = {'tok': round(float(s[0]), 4), 'cross': round(float(s[1]), 4), 'dev': round(float(s[2]), 4)}
        print(f"L{L} HELD-OUT variance shares: {var_shares[str(L)]}", flush=True)
        del X

    # CE substitutions on B
    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['layer'] = -1; base = ce(B)
    ce_res = {}
    for L in LAYERS:
        row = {}
        for mode in ['full', 'tok', 'dev', 'meanabl']:
            SUB['layer'] = L; SUB['mode'] = mode
            row[mode] = round(ce(B) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        ce_res[str(L)] = row
        print(f"L{L} HELD-OUT CE: full {row['full']} | tok {row['tok']} | dev-only {row['dev']} | mean-abl {row['meanabl']}", flush=True)
    for h in hs: h.remove()

    insample = {'1': {'tok': 0.6576}, '8': {'tok': 0.3255}, '16': {'tok': 0.9062}}
    out = {'base_ce': round(base, 4), 'var_shares': var_shares, 'ce': ce_res, 'insample_tok_share': insample}
    v16 = var_shares['16']['tok']
    out['pred_a_readout_reinstated'] = bool(v16 > 0.7 and ce_res['16']['dev'] > 2*ce_res['16']['meanabl'])
    out['pred_a_readout_withdrawn_final'] = bool(v16 < 0.5)
    out['pred_b_deep_context'] = bool(var_shares['8']['tok'] < 0.3255 and var_shares['8']['dev'] > 0.4187)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L16 held-out tok share {v16} (in-sample 0.906) | reinstate {out['pred_a_readout_reinstated']} | withdraw-final {out['pred_a_readout_withdrawn_final']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
