# repeat_quad_sufficient: are the four reader heads SUFFICIENT, not just necessary?
#
# §1207 (repeat_heads2): masking the quad's long-range reads (L2H5, L3H8, L8H3, L8H4) costs
# 2.20 of all18's 3.20 (69%) — necessity. The converse condition: mask EVERY head's reads to
# W=64 EXCEPT the quad (they keep full context). If the model then recovers most of the
# repeat performance, four nameable heads are the copy circuit's entire long-range front end
# — reading channel certified both directions.
#
# Conditions: base; quad (replication anchor); allbutquad (all 162 heads masked except the
# four); all18. Rows/instrument as §1204-07 (W=64, pos-0 visible, repeat rows, t>=128).
#
# Registered predictions:
#   pred_a QUAD SUFFICES: cost(allbutquad) <= 0.3 x cost(all18) — keeping just 4 heads'
#          range restores >= 70% of the repeat capability.
#   pred_b COMPLEMENTARITY: cost(quad) + cost(allbutquad) is within [0.8, 1.3] x cost(all18)
#          — the split is a genuine partition of the long-range function (approximate
#          additivity of the two complementary masks).
#   pred_c REPLICATION: cost(quad) within ±0.05 of §1207's 2.2029.
# Controls: sanity base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_quad_sufficient_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None
ALLH = set(range(9))
QUAD = {2: {5}, 3: {8}, 8: {3, 4}}


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
    abq = {L: (ALLH - QUAD[L]) if L in QUAD else ALLH for L in range(18)}
    CONDS = {'base': {}, 'quad': QUAD, 'allbutquad': abq,
             'all18': {L: ALLH for L in range(18)}}
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
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_quad_suffices': bool(cost['allbutquad'] <= 0.3 * cost['all18']),
           'pred_b_partition': bool(0.8 * cost['all18'] <= cost['quad'] + cost['allbutquad'] <= 1.3 * cost['all18']),
           'pred_c_replicates': bool(abs(cost['quad'] - 2.2029) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a suffices {out['pred_a_quad_suffices']} | pred_b partition {out['pred_b_partition']} | pred_c repl {out['pred_c_replicates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
