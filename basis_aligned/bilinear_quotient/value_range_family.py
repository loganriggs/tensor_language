# value_range_family: §1186's band map on swiglu18 (family symmetry of the pooling channel).
# Read-mask attention beyond W=64 (pos-0 visible) one band at a time; softmax masking = -inf
# BEFORE softmax. Registered: (a) mid1 (L5-9) is the top band; (b) late (15-17) <= 0.03;
# (c) all18 within 2x of bilin18's 0.1759.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'value_range_family_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_banded(idx, band, MASK_W, FULL):
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
        msk = MASK_W if L in band else FULL
        scores = scores.masked_fill(~msk, float('-inf'))
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
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    BANDS = {'base': set(), 'front': set(range(5)), 'mid1': set(range(5, 10)),
             'mid2': set(range(10, 15)), 'late': {15, 16, 17}, 'all18': set(range(18))}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in BANDS}; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        for cname, band in BANDS.items():
            lo = forward_banded(idx, band, MASK_W, FULL).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction="sum"))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {c: round(CE[c] - CE['base'], 4) for c in BANDS if c != 'base'}
    bandvals = {c: cost[c] for c in ('front', 'mid1', 'mid2', 'late')}
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'bilin18_refs': {'front': 0.0358, 'mid1': 0.067, 'mid2': 0.0335, 'late': 0.013, 'all18': 0.1759},
           'pred_a_mid1_top': bool(cost['mid1'] == max(bandvals.values())),
           'pred_b_late_local': bool(cost['late'] <= 0.03),
           'pred_c_all18_2x': bool(cost['all18'] <= 2 * 0.1759),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"pred_a mid1-top {out['pred_a_mid1_top']} | pred_b late {out['pred_b_late_local']} | pred_c 2x {out['pred_c_all18_2x']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
