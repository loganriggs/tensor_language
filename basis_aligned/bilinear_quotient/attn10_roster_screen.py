# attn10_roster_screen: COMPLETE THE a10 ROSTER (flagged S1448: hybrid gain only +.21
# with roster {2,5,6}; composite2 shows a10's live roster HELPS in composite, so a
# fuller roster is leverage). Per-layer arm: ONLY layer 10 kernelized, roster heads
# live; add each remaining head {0,1,3,4,7,8} singly to {2,5,6}, then greedy add the
# best two. Scored as fid_opt vs the FROZEN attn10 anchor (198-sweep). NR=960.
#
# Registered predictions:
#   pred_a >= 1 single addition adds >= .05 fid over roster {2,5,6}.
#   pred_b best 4-head roster >= .70 fid (current 3-head = .598).
#   pred_c greedy 5-head roster >= .75 fid.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn10_roster_screen_results.json'
NMEAN = 24; NR = 960; LT = 10
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
BASE = (2, 5, 6)
KERN = {}


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
def fwd_arm(idx, live):
    """live=None -> clean; else layer LT kernel + heads in live keep their pattern."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        pat = block_pat(at, xin, B)
        if live is not None and L == LT:
            newp = KERN['k'].unsqueeze(0).expand(B, -1, -1, -1).to(pat.dtype).clone()
            for hh in live:
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

    ACC = torch.zeros(9, T, T); nb = 0
    for i in range(0, NMEAN, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            pat = block_pat(at, xin, B)
            if L == LT:
                ACC += pat.float().mean(0).cpu()
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        nb += 1
    mp = (ACC / nb).to(DEV)
    kern = torch.zeros_like(mp)
    for d_ in range(T):
        idxs = torch.arange(d_, T)
        kern[:, idxs, idxs - d_] = mp[:, idxs, idxs - d_].mean(1).unsqueeze(1)
    KERN['k'] = kern
    print("kernel cached", flush=True)

    def ce_run(live):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, live).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    sw = json.load(open(PT + 'optimal_ablation_all_results.json'))['results']['attn10']
    clean = ce_run(None)
    fid = lambda ce_: (sw['ce_opt'] - ce_) / max(sw['ce_opt'] - clean, 1e-6)
    res = {'clean': round(clean, 4)}
    base_ce = ce_run(BASE)
    res['base_256'] = round(base_ce, 4)
    print(f"clean {clean:.4f} base {base_ce:.4f} fid {fid(base_ce):.4f}", flush=True)

    singles = {}
    for h in (0, 1, 3, 4, 7, 8):
        ce_ = ce_run(BASE + (h,))
        singles[h] = ce_
        res[f'add_{h}'] = round(ce_, 4)
        print(f"add {h}: {ce_:.4f} fid {fid(ce_):.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    order = sorted(singles, key=lambda h: singles[h])
    b1 = order[0]
    ce5 = ce_run(BASE + (b1, order[1]))
    res[f'greedy5_{b1}_{order[1]}'] = round(ce5, 4)

    fids = {k: round(fid(v), 4) for k, v in
            [('base', base_ce)] + [(f'add_{h}', singles[h]) for h in singles]
            + [('greedy5', ce5)]}
    best_gain = max(fid(singles[h]) - fid(base_ce) for h in singles)
    pa = best_gain >= 0.05
    pb = fid(singles[b1]) >= 0.70
    pc = fid(ce5) >= 0.75
    out = {'ce': res, 'fid_opt': fids, 'best_single_add': b1,
           'greedy_pair': [b1, order[1]], 'best_single_gain': round(best_gain, 4),
           'frozen_anchor': {'ce_mean': sw['ce_mean'], 'ce_opt': sw['ce_opt']},
           'pred_a_single_adds_05': bool(pa), 'pred_b_4head_70': bool(pb),
           'pred_c_5head_75': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"fids {fids}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
