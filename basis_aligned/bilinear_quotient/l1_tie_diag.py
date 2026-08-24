# l1_tie_diag: is L1's low argmax hit (0.68, the 162-head map's only laggard, §1165) just
# TIE-INSTABILITY among near-equal scores? §1168 showed it costs 0.0005 nats — loss-irrelevant.
#
# For L1's 9 heads at W=128 (exact-prefix window fold): top-1 hit, top-3 hit, Pearson
# correlation between folded and real pattern rows, and each head's TIE RATE (fraction of
# query positions where the real top-2 scores differ by < 10% of the top score's magnitude).
#
# Registered predictions:
#   pred_a TIES EXPLAIN IT: L1 mean top-3 hit >= 0.85 (vs top-1 0.68).
#   pred_b PATTERNS MATCH: mean folded-vs-real row correlation >= 0.9.
#   pred_c TIE RATE PREDICTS THE LAGGARDS: the 3 lowest-top1 heads have higher mean tie rate
#          than the 3 highest-top1 heads.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l1_tie_diag_results.json'
NR = 16; W = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def window_resid(tokens, W, nblocks):
    B, Tn = tokens.shape
    idx = torch.arange(Tn, device=DEV)
    win = torch.stack([tokens[:, (idx + o).clamp_min(0)] for o in range(-(W - 1), 1)], -1)
    flat = win.reshape(B * Tn, W)
    outs = []
    step = max(128, 4096 // W)
    for i in range(0, flat.shape[0], step):
        wb = flat[i:i + step]
        x = F.rms_norm(m.transformer.wte(wb), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h[:nblocks]:
            x, v1 = blk(x, v1, x0)
        outs.append(x[:, -1].detach())
    res = torch.cat(outs, 0).reshape(B, Tn, D)
    Wp = min(W, Tn)
    xp = F.rms_norm(m.transformer.wte(tokens[:, :Wp]), (D,)); x0p = xp; v1p = None
    for blk in m.transformer.h[:nblocks]:
        xp, v1p = blk(xp, v1p, x0p)
    res[:, :Wp] = xp.detach()
    return res


@torch.no_grad()
def all_head_patterns(X, li, cos, sin):
    at = m.transformer.h[li].attn
    B = X.shape[0]
    qf = F.rms_norm(at.c_q(X).view(B, T, 9, 128), (128,))
    kf = F.rms_norm(at.c_k(X).view(B, T, 9, 128), (128,))
    q2 = F.rms_norm(at.c_q2(X).view(B, T, 9, 128), (128,))
    k2 = F.rms_norm(at.c_k2(X).view(B, T, 9, 128), (128,))
    qf = are(qf, cos, sin); kf = are(kf, cos, sin)
    q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', qf.float(), kf.float())
           * torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()))
    return pat * torch.tril(torch.ones(T, T, device=DEV))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    L = 1
    agg = {h: {'top1': 0, 'top3': 0, 'tie': 0, 'n': 0, 'corr': []} for h in range(9)}
    cap = {}
    def pre(mo_, args): cap['x'] = args[0]
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        hk = m.transformer.h[L].attn.register_forward_pre_hook(pre)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        hk.remove()
        at = m.transformer.h[L].attn
        cos, sin = at.rotary(at.c_q(cap['x']).view(4, T, 9, 128))
        rp = all_head_patterns(cap['x'], L, cos, sin)
        xh = window_resid(idx, W, L)
        fp = all_head_patterns(xh, L, cos, sin)
        for h in range(9):
            for b in range(4):
                for q in range(128, T, 5):
                    r = rp[b, h, q, :q].abs(); f = fp[b, h, q, :q].abs()
                    kr = int(r.argmax()); kf1 = int(f.argmax())
                    top3 = torch.topk(f, 3).indices.tolist()
                    v2 = torch.topk(r, 2).values
                    st = agg[h]
                    st['top1'] += int(kr == kf1); st['top3'] += int(kr in top3)
                    st['tie'] += int(float(v2[0] - v2[1]) < 0.1 * float(v2[0].abs() + 1e-9))
                    st['n'] += 1
                    rr = rp[b, h, q, :q]; ff = fp[b, h, q, :q]
                    c = torch.corrcoef(torch.stack([rr, ff]))[0, 1]
                    if torch.isfinite(c): st['corr'].append(float(c))
    stats = {}
    for h in range(9):
        st = agg[h]; n = max(st['n'], 1)
        stats[str(h)] = {'top1': round(st['top1'] / n, 3), 'top3': round(st['top3'] / n, 3),
                         'tie_rate': round(st['tie'] / n, 3),
                         'corr': round(sum(st['corr']) / max(len(st['corr']), 1), 3)}
    t1 = [stats[str(h)]['top1'] for h in range(9)]
    order = sorted(range(9), key=lambda h: t1[h])
    low3 = sum(stats[str(h)]['tie_rate'] for h in order[:3]) / 3
    high3 = sum(stats[str(h)]['tie_rate'] for h in order[-3:]) / 3
    mtop3 = sum(stats[str(h)]['top3'] for h in range(9)) / 9
    mcorr = sum(stats[str(h)]['corr'] for h in range(9)) / 9
    out = {'n_rows': NR, 'per_head': stats,
           'mean_top3': round(mtop3, 3), 'mean_corr': round(mcorr, 3),
           'tie_rate_low3_vs_high3': [round(low3, 3), round(high3, 3)],
           'pred_a_ties_explain': bool(mtop3 >= 0.85),
           'pred_b_patterns_match': bool(mcorr >= 0.9),
           'pred_c_tie_predicts': bool(low3 > high3),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for h in range(9):
        print(f"L1H{h}: {stats[str(h)]}", flush=True)
    print(f"mean top3 {out['mean_top3']} | mean corr {out['mean_corr']} | tie low3 {low3:.3f} vs high3 {high3:.3f}")
    print(f"pred_a ties {out['pred_a_ties_explain']} | pred_b corr {out['pred_b_patterns_match']} | pred_c tie-predicts {out['pred_c_tie_predicts']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
