# attn_composite: THE HEADLINE COMPOSITE, ATTENTION SIDE (S1447-48: kernels describe
# the generic stack, the named roster the rest). ALL 18 layers simultaneously:
#   kernel_all — every head at every layer on its distance kernel (values live).
#   hybrid_all — kernels everywhere + the S1448 roster live at layers 5/8/10/13/14/16/17.
# MLPs live. Global CE, mask >= 64, NR=960. This is the spec's nightly-composite shape:
# errors COMPOUND across layers, so per-layer fids do not predict this number — that is
# the point of measuring it.
#
# Registered predictions:
#   pred_a kernel_all composite costs <= 1.2 CE over clean (compounding bounded).
#   pred_b hybrid_all recovers >= .60 of the kernel_all -> clean gap.
#   pred_c hybrid_all lands <= clean + 0.5 CE.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}


@torch.no_grad()
def fwd_arm(idx, LT, mode, meanpat, kern):
    """LT unused in composite mode; kept for signature stability."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        if mode is not None:
            kk2 = KERNS[L]
            newp = kk2.unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            if mode == 'hybrid_all' and L in SPEC:
                for hh in SPEC[L]:
                    newp[:, hh] = pat[:, hh]
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MEANR = cl.fineweb_rows(NMEAN, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    # ONE capture pass for every layer's mean pattern
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
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            ACC[L] += pat.float().mean(0).cpu()
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    MP = {}
    for L in range(18):
        mp = (ACC[L] / nb).to(DEV)
        kern = torch.zeros_like(mp)
        for d_ in range(T):
            idxs = torch.arange(d_, T)
            kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
        MP[L] = mp; MP[str(L) + 'k'] = kern
    print("all 18 kernels cached", flush=True)

    global KERNS
    KERNS = {L: MP[str(L) + 'k'] for L in range(18)}

    def ce_run(LT, mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT, mode, None, None).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    clean = ce_run(1, None)
    print(f"clean {clean:.4f}", flush=True)
    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']
    res = {'clean': round(clean, 4)}
    fids = {}
    ka = ce_run(0, 'kernel_all')
    print(f"kernel_all: {ka:.4f} (clean {clean:.4f})", flush=True)
    hy = ce_run(0, 'hybrid_all')
    print(f"hybrid_all: {hy:.4f}", flush=True)
    d_k = ka - clean; d_h = hy - clean
    rec = (ka - hy) / max(ka - clean, 1e-6)
    pa = d_k <= 1.2
    pb = rec >= 0.60
    pc = d_h <= 0.5
    out = {'clean': round(clean, 4), 'kernel_all': round(ka, 4),
           'hybrid_all': round(hy, 4), 'delta_kernel': round(d_k, 4),
           'delta_hybrid': round(d_h, 4), 'roster_recovery': round(rec, 4),
           'pred_a_kernel_le_12': bool(pa), 'pred_b_roster_60': bool(pb),
           'pred_c_hybrid_le_05': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"dk {d_k:.4f} dh {d_h:.4f} rec {rec:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
