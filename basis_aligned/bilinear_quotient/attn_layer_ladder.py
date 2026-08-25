# attn_layer_ladder: FIRST STAND-INS FOR WHOLE ATTENTION LAYERS (priority board tops:
# attn4 .223 / attn1 .219 unexplained, no stand-ins ever). Arms per target layer
# (attn1 and attn4), scored on FROZEN sweep anchors, global CE, mask >= 64:
#   meanpat  — every head's attention pattern replaced by its 24-row mean [9,T,T]
#              (values live): prices "where to look" as a static positional table
#              (9xTxT x16 = 9.4 Mbit).
#   distkern — pattern replaced by a per-head DISTANCE KERNEL (mean pattern value as a
#              function of query-key offset, derived from meanpat; 9xT params = 37 Kbit;
#              lower triangle only).
# Values stay live in both (value maps are the module's own weights — a stand-in for
# them is the next rung). Assumptions registered. Fit rows skip=80 (means), EVAL
# skip=7000, NR=960.
#
# Registered predictions:
#   pred_a attn1 meanpat >= .60 fid_opt (front patterns are generic/positional).
#   pred_b attn1 distkern >= .8x its meanpat (distance carries most of the pattern).
#   pred_c attn4 preserves the ordering (meanpat >= distkern) with meanpat >= .50.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_layer_ladder_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
TARGETS = (1, 4)


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
            if mode == 'meanpat':
                pat = meanpat.unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype)
            else:  # distkern
                pat = kern.unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype)
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

    # capture mean patterns for the target layers
    MP = {}
    for LT in TARGETS:
        caps = []
        def cap_hook(mod, args):
            return None
        # capture via a manual pass
        acc = torch.zeros(9, T, T)
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
                if L == LT:
                    acc += pat.float().mean(0).cpu(); nb += 1
                v = at.c_v(xin).view(B, T, 9, 128)
                if v1 is None:
                    v1 = v
                vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
                x = xm + at.c_proj(y.reshape(B, T, D))
                x = x + blk.mlp(F.rms_norm(x, (D,)))
        MP[LT] = (acc / nb).to(DEV)
        # distance kernel: mean of meanpat over (q-k) offset, lower triangle
        kern = torch.zeros_like(MP[LT])
        mp = MP[LT]
        for d_ in range(T):
            idxs = torch.arange(d_, T)
            vals = mp[:, idxs, idxs - d_].mean(1)          # [9]
            kern[:, idxs, idxs - d_] = vals.unsqueeze(1)
        MP[str(LT) + 'k'] = kern
        print(f"attn{LT} patterns cached", flush=True)

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
        for mode in ('meanpat', 'distkern'):
            ce_ = ce_run(LT, mode)
            f = (anchor - ce_) / max(anchor - clean, 1e-6)
            res[f'attn{LT}_{mode}'] = round(ce_, 4)
            fids[f'attn{LT}_{mode}'] = round(f, 4)
            print(f"attn{LT} {mode}: CE {ce_:.4f} fid {f:.4f}", flush=True)
            json.dump({'partial': True, 'res': res, 'fids': fids}, open(OUT, 'w'), indent=1)

    pa = fids['attn1_meanpat'] >= 0.60
    pb = fids['attn1_distkern'] >= 0.8 * max(fids['attn1_meanpat'], 1e-6)
    pc = fids['attn4_meanpat'] >= fids['attn4_distkern'] and fids['attn4_meanpat'] >= 0.50
    out = {'ce': res, 'fid_opt': fids,
           'mbits': {'meanpat': round(9 * T * T * 16 / 1e6, 2),
                     'distkern': round(9 * T * 16 / 1e6, 3)},
           'pred_a_a1_meanpat_60': bool(pa), 'pred_b_a1_kernel_carries': bool(pb),
           'pred_c_a4_ordering': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
