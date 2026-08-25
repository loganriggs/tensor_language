# attn_composite2: WHY DID THE LIVE ROSTER HURT? (S1449: hybrid_all 5.28 WORSE than
# kernel_all 4.73, recovery -0.31.) Hypothesis: roster heads compute q/k from a stream
# corrupted by 17 kernelized layers, so their LIVE patterns misfire; data-independent
# kernels are robust to upstream perturbation, data-dependent subcomponents are not.
# Arms (all vs kernel_all reference, NR=960, mask >= 64):
#   hybrid_frozen — kernels everywhere + roster heads use patterns captured from a
#                   CLEAN pass on the same rows (teacher patterns). If corruption is
#                   the cause, this repairs what live computation could not.
#   marg_L (7 arms) — kernel_all but roster LIVE at only layer L. Locates offenders.
#
# Registered predictions:
#   pred_a hybrid_frozen recovers >= .30 of the kernel_all -> clean gap.
#   pred_b layer 5's marginal arm (live sink head 5.7) has the highest CE of the 7.
#   pred_c at least 2 of 7 marginal arms land ABOVE kernel_all (live roster harms
#          even one layer at a time in composite context).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite2_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}
KERNS = {}


@torch.no_grad()
def block_pat(at, xin, B):
    cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
    q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
    q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat.masked_fill(~tril, 0.0)


@torch.no_grad()
def fwd_full(idx, replace_layers=None, live_layers=frozenset(), frozen=None):
    """replace_layers: set of layers whose pat -> kernel (roster heads exempted per
    live_layers/frozen). live_layers: layers where SPEC heads stay live. frozen:
    dict {L: clean pat tensor} — SPEC heads at those layers use the clean pattern."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(at, xin, B)
        if replace_layers is not None and L in replace_layers:
            newp = KERNS[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            if L in SPEC:
                if L in live_layers:
                    for hh in SPEC[L]:
                        newp[:, hh] = pat[:, hh]
                elif frozen is not None and L in frozen:
                    for hh in SPEC[L]:
                        newp[:, hh] = frozen[L][:, hh].to(pat.dtype)
            pat = newp
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def clean_pats(idx):
    """One clean pass; return {L: pat} for roster layers (full 9-head pat kept)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; store = {}
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(at, xin, B)
        if L in SPEC:
            store[L] = pat
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return store


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MEANR = cl.fineweb_rows(NMEAN, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    ACC = {L: torch.zeros(9, T, T) for L in range(18)}
    nb = 0
    for i in range(0, NMEAN, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            ACC[L] += pat.float().mean(0).cpu()
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    for L in range(18):
        mp = (ACC[L] / nb).to(DEV)
        kern = torch.zeros_like(mp)
        for d_ in range(T):
            idxs = torch.arange(d_, T)
            kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
        KERNS[L] = kern
    print("kernels cached", flush=True)

    ALL = set(range(18))

    def ce_run(kind, arg=None):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if kind == 'clean':
                lo = fwd_full(idx)
            elif kind == 'kernel_all':
                lo = fwd_full(idx, ALL)
            elif kind == 'frozen':
                fz = clean_pats(idx)
                lo = fwd_full(idx, ALL, frozen=fz)
            elif kind == 'marg':
                lo = fwd_full(idx, ALL, live_layers=frozenset({arg}))
            lo = lo.float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {}
    res['clean'] = round(ce_run('clean'), 4)
    print(f"clean {res['clean']}", flush=True)
    res['kernel_all'] = round(ce_run('kernel_all'), 4)
    print(f"kernel_all {res['kernel_all']}", flush=True)
    res['hybrid_frozen'] = round(ce_run('frozen'), 4)
    print(f"hybrid_frozen {res['hybrid_frozen']}", flush=True)
    for L in sorted(SPEC):
        res[f'marg_{L}'] = round(ce_run('marg', L), 4)
        print(f"marg_{L} {res[f'marg_{L}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    gap = res['kernel_all'] - res['clean']
    rec_f = (res['kernel_all'] - res['hybrid_frozen']) / max(gap, 1e-6)
    margs = {L: res[f'marg_{L}'] for L in sorted(SPEC)}
    worst = max(margs, key=lambda L: margs[L])
    n_harm = sum(1 for L in margs if margs[L] > res['kernel_all'])
    pa = rec_f >= 0.30
    pb = worst == 5
    pc = n_harm >= 2
    out = {'ce': res, 'gap_kernel': round(gap, 4), 'rec_frozen': round(rec_f, 4),
           'marginal_vs_kernel': {str(L): round(margs[L] - res['kernel_all'], 4)
                                  for L in sorted(margs)},
           'worst_marginal_layer': worst, 'n_marginals_harmful': n_harm,
           'pred_a_frozen_rec_30': bool(pa), 'pred_b_worst_is_5': bool(pb),
           'pred_c_2_harmful': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"rec_frozen {rec_f:.4f} worst marg_{worst} n_harm {n_harm}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
