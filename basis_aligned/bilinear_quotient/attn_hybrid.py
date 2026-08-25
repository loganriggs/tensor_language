# attn_hybrid: KERNEL + NAMED SPECIALISTS LIVE (S1447: content layers collapse under
# pure kernels — the named heads ARE the kernel-resistance). For each content layer,
# generic heads get the distance kernel, the dossier's specialists keep their LIVE
# patterns (values live everywhere). Scored on FROZEN anchors. Fit skip=80, EVAL
# skip=7000, NR=960.
#
# Registered predictions:
#   pred_a attn5 hybrid >= .60 (the sink head 5.7 was the whole problem).
#   pred_b every hybrid gains >= .15 over its S1447 kernel-all fid.
#   pred_c attn13 hybrid >= .50.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_hybrid_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
TARGETS = (5, 8, 10, 13, 14, 16, 17)
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}
KERNEL_FID = {5: -0.055, 8: 0.0398, 10: 0.3907, 13: 0.042, 14: 0.0555,
              16: 0.1537, 17: -0.0291}


@torch.no_grad()
def fwd_arm(idx, LT, mode, meanpat, kern):
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
        if L == LT and mode is not None:
            newp = kern.unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            for hh in SPEC[LT]:
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

    def ce_run(LT, mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, LT, mode, MP.get(LT), MP.get(str(LT) + 'k')).float()
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
    for LT in TARGETS:
        anchor = sw[f'attn{LT}']['ce_opt']
        ce_ = ce_run(LT, 'hybrid')
        f = (anchor - ce_) / max(anchor - clean, 1e-6)
        res[f'attn{LT}'] = round(ce_, 4)
        fids[f'attn{LT}'] = round(f, 4)
        print(f"attn{LT} hybrid: CE {ce_:.4f} fid {f:.4f} "
              f"(kernel-all was {KERNEL_FID[LT]:+.3f})", flush=True)
        json.dump({'partial': True, 'res': res, 'fids': fids}, open(OUT, 'w'), indent=1)

    gains = {L: fids[f'attn{L}'] - KERNEL_FID[L] for L in TARGETS}
    pa = fids['attn5'] >= 0.60
    pb = all(g >= 0.15 for g in gains.values())
    pc = fids['attn13'] >= 0.50
    out = {'ce': res, 'fid_opt': fids, 'kernel_ref': KERNEL_FID,
           'gains': {str(L): round(g, 4) for L, g in gains.items()},
           'pred_a_a5_60': bool(pa), 'pred_b_all_gain_15': bool(pb),
           'pred_c_a13_50': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gains {out['gains']}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
