# repeat_family_prose: swiglu18's double-duty row — do ITS copy stations (L5H2, L4H4,
# L8H0+H8) also participate in ordinary prose pooling, like bilin18's quad (22%, §1211)?
#
# Instrument: swiglu18 per-head read-masks (softmax -inf), W=64, pos-0 visible, NATURAL
# FineWeb rows (no repeat), scored t>=128. Conditions: base; quad {5.2, 4.4, 8.0, 8.8};
# allbutquad; all18.
#
# Registered predictions:
#   pred_a DOUBLE DUTY IS FAMILY-GENERAL: 0.05 x cost(all18) <= cost(quad) <= 0.4 x
#          cost(all18) (participates, but far below its copy-regime share).
#   pred_b REMAINDER CARRIES PROSE: cost(allbutquad) >= 0.7 x cost(all18).
#   pred_c PARTITION: cost(quad) + cost(allbutquad) within [0.8, 1.3] x cost(all18).
# Control: all18 must land near swiglu18's §1188 prose read-mask 0.2231 (±0.03).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_family_prose_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL = None
ALLH = set(range(9))


@torch.no_grad()
def forward_headmasked(idx, spec):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(q, cos, sin); k = are(k, cos, sin)
        scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        heads = spec.get(L, None)
        if heads is None:
            scores = scores.masked_fill(~FULL, float('-inf'))
        else:
            msk = torch.stack([MASK_W if h in heads else FULL for h in range(9)], 0)
            scores = scores.masked_fill(~msk.unsqueeze(0), float('-inf'))
        pat = F.softmax(scores, dim=-1)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    QUAD = {4: {4}, 5: {2}, 8: {0, 8}}
    ABQ = {L: (ALLH - QUAD[L]) if L in QUAD else ALLH for L in range(18)}
    CONDS = {'base': {}, 'quad': QUAD, 'allbutquad': ABQ,
             'all18': {L: ALLH for L in range(18)}}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        for cname, spec in CONDS.items():
            lo = forward_headmasked(idx, spec).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'near_1188': bool(abs(cost['all18'] - 0.2231) <= 0.03),
           'pred_a_double_duty': bool(0.05 * cost['all18'] <= cost['quad'] <= 0.4 * cost['all18']),
           'pred_b_remainder': bool(cost['allbutquad'] >= 0.7 * cost['all18']),
           'pred_c_partition': bool(0.8 * cost['all18'] <= cost['quad'] + cost['allbutquad'] <= 1.3 * cost['all18']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"near1188 {out['near_1188']} | pred_a duty {out['pred_a_double_duty']} | pred_b remainder {out['pred_b_remainder']} | pred_c partition {out['pred_c_partition']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
