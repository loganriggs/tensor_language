# attn_composite3: BUILD THE LEGITIMATE BEST COMPOSITE from the S1450 marginal map.
# Teacher patterns are DIAGNOSTIC ONLY (they run the full model — not a stand-in).
# The legitimate moves the marginals license:
#   - live roster HELPS singly at layers 10/13/14/16/17 (marginals -.02..-.13),
#     HARMS at 5 (+.68, live sink 5.7 on corrupted stream) and 8 (+.11) -> drop those.
#   - head 5.7 is the known bias-head (S1089: one fixed vector ~ .985 local fid);
#     its frozen OPTIMAL CONSTANT (c_proj-input slice from the 198-sweep) is a priced,
#     data-independent stand-in — use it INSTEAD of the kernel at 5.7.
# Arms (NR=960, mask >= 64, MLPs/values live):
#   best_roster  — kernel_all + live roster only at {10,13,14,16,17}.
#   sink_const   — best_roster + head 5.7's y-slice = its optimal constant.
#   sink_a8const — sink_const + heads 8.1/8.2/8.3/8.7 y-slices = their opt constants.
#
# Registered predictions:
#   pred_a best_roster <= 4.45 CE (marginals roughly additive: 4.73 - .33).
#   pred_b sink_const improves >= .10 CE over best_roster (bias-head constant fixes
#          what the kernel could not at the sink layer).
#   pred_c sink_a8const <= 4.20 CE (beats the teacher-pattern diagnostic ceiling 4.15
#          is NOT claimed; 4.20 is the bar).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_composite3_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
SPEC = {5: {7}, 8: {1, 2, 3, 7}, 10: {2, 5, 6}, 13: {0, 5, 8},
        14: {4, 6, 7}, 16: {0, 3, 4, 5}, 17: {0, 1, 2}}
LIVE_BEST = frozenset({10, 13, 14, 16, 17})
KERNS = {}
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
C57 = CONSTS['head5.7'].to(DEV).float()
C8 = {h: CONSTS[f'head8.{h}'].to(DEV).float() for h in (1, 2, 3, 7)}


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
def fwd_full(idx, mode):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(at, xin, B)
        if mode is not None:
            newp = KERNS[L].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            if L in SPEC and L in LIVE_BEST:
                for hh in SPEC[L]:
                    newp[:, hh] = pat[:, hh]
            pat = newp
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if mode in ('sink_const', 'sink_a8const') and L == 5:
            y[:, :, 7, :] = C57.to(y.dtype)
        if mode == 'sink_a8const' and L == 8:
            for hh, cv in C8.items():
                y[:, :, hh, :] = cv.to(y.dtype)
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


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

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_full(idx, mode).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {}
    for mode in [None, 'best_roster', 'sink_const', 'sink_a8const']:
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    br = res['best_roster']; sc = res['sink_const']; sa = res['sink_a8const']
    pa = br <= 4.45
    pb = (br - sc) >= 0.10
    pc = sa <= 4.20
    out = {'ce': res, 'gain_sink_const': round(br - sc, 4),
           'gain_a8_const': round(sc - sa, 4),
           'pred_a_best_le_445': bool(pa), 'pred_b_sink_gain_10': bool(pb),
           'pred_c_final_le_420': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"best {br} sink {sc} a8 {sa}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
