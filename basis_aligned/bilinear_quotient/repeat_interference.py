# repeat_interference: WHY does masking a station's sibling heads WITH it recover loss?
#
# Three cross-model observations (§1205/§1207/§1210): masking ONLY the fetcher head costs
# MORE than masking its whole layer (L2H5 0.62 > L2 0.53; L3H8 1.076 > L3 1.008; swiglu
# L4H4 0.66 > L4 0.40). Hypothesis space: with the fetcher blinded, some sibling head's
# long-range read injects actively-harmful (stale/mismatched) values that the downstream
# circuit misreads as copy evidence. This locates WHICH sibling(s) at bilin18's L3.
#
# Conditions (repeat rows, W=64 read-mask, pos-0 visible): base; L3H8 (anchor); L3 (anchor);
# L3H8 plus each sibling h in {0..8}\{8} masked jointly (8 conditions) — if one sibling
# accounts for the recovery, its joint condition drops to ~L3's 1.008.
#
# Registered predictions:
#   pred_a CONCENTRATED: one sibling accounts for >= 60% of the recovery — i.e. min over
#          siblings of cost(L3H8+h) <= cost(L3H8) − 0.6 x (cost(L3H8) − cost(L3)).
#   pred_b NO SIBLING HELPS ALONE: for every sibling, cost(L3H8+h) <= cost(L3H8) + 0.02
#          (adding masks never makes things WORSE than the fetcher-only condition).
#   pred_c ANCHORS REPLICATE §1207 (±0.05).
# Control: sanity base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_interference_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None
ALLH = set(range(9))


def make_mask():
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


@torch.no_grad()
def forward_headmasked(idx, spec):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        q2 = F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,))
        k2 = F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,))
        q = are(q, cos, sin); k = are(k, cos, sin); q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        heads = spec.get(L, None)
        if heads is None:
            pat = pat.masked_fill(~FULL_TRIL, 0.0)
        else:
            msk = torch.stack([MASK_W if h in heads else FULL_TRIL for h in range(9)], 0)
            pat = pat.masked_fill(~msk.unsqueeze(0), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    global MASK_W, FULL_TRIL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MASK_W = make_mask()
    FULL_TRIL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': {}, 'L3H8': {3: {8}}, 'L3': {3: ALLH}}
    for h in range(8):
        CONDS[f'L3H8p{h}'] = {3: {8, h}}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, spec in CONDS.items():
            lo = forward_headmasked(idx, spec).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    rec_full = cost['L3H8'] - cost['L3']
    sib = {h: cost[f'L3H8p{h}'] for h in range(8)}
    best = min(sib, key=sib.get)
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'recovery_full': round(rec_full, 4),
           'best_sibling': best, 'best_cost': sib[best],
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_concentrated': bool(sib[best] <= cost['L3H8'] - 0.6 * rec_full),
           'pred_b_never_worse': bool(all(v <= cost['L3H8'] + 0.02 for v in sib.values())),
           'pred_c_replicates': bool(abs(cost['L3H8'] - 1.0755) <= 0.05 and
                                     abs(cost['L3'] - 1.0076) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"recovery {out['recovery_full']} | best sibling H{best} at {sib[best]}")
    print(f"sanity {out['sanity']} | pred_a conc {out['pred_a_concentrated']} | pred_b never-worse {out['pred_b_never_worse']} | pred_c repl {out['pred_c_replicates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
