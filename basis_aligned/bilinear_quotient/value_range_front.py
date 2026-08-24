# value_range_front: §1188 front drill-down — L0-4 singles @W64 (pos-0 visible) + joint.
# Registered: (a) L1+L2 carry the majority of the front joint (router/trigger band);
# (b) singles sum <= front joint (super-additive); (c) L0 <= 0.003 (bigram table, no range).
#
# value_range_bands: WHERE does the 0.07-nat long-range value-pooling live? (§1185 thread.)
#
# The window-certificate program (§1161-85) leaves exactly one long-range channel: attention
# VALUE POOLING (whole-model window cost 0.082@128 / 0.207@64, of which selection is only
# 0.014/0.067). This experiment window-restricts attention READS — the pattern masked beyond
# W=64 — one depth band at a time, with POSITION 0 ALWAYS VISIBLE (writeup 483: the sink's
# constant fetch is the known, already-priced non-local read; masking it would just re-measure
# that). Everything else (MLPs, values within window, other bands) stays full-context.
#
# Conditions: base; front (L0-4); mid1 (L5-9); mid2 (L10-14); late (L15-17); all18.
# Scored positions >= 128 (same convention as §1180).
#
# Registered predictions:
#   pred_a MID CARRIES IT: cost(mid1) + cost(mid2) >= 0.6 × cost(all18) — the content pool
#          is gathered in the deep-middle (§1076 value-residual bag + §1099 middle pool).
#   pred_b LATE IS LOCAL: cost(late) <= 0.02 (readout consumes own-position coords, §1153).
#   pred_c SUPER-ADDITIVE: sum of the four band costs <= cost(all18) is FALSE — i.e., bands
#          sum to LESS than the joint (redundant carriage, §1160-style); quantify the factor.
# Control: base = unmasked custom forward, must equal true-model CE (sanity, ±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'value_range_front_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


def make_mask():
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)                       # position 0 always visible
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


MASK_W = None
FULL_TRIL = None


@torch.no_grad()
def forward_banded(idx, band):
    """Full model; attention pattern masked to the WIN-window (+pos 0) for layers in band."""
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
        msk = MASK_W if L in band else FULL_TRIL
        pat = pat.masked_fill(~msk, 0.0)
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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    BANDS = {'base': set(), 'L0': {0}, 'L1': {1}, 'L2': {2}, 'L3': {3}, 'L4': {4},
             'front': set(range(5))}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in BANDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, band in BANDS.items():
            lo = forward_banded(idx, band).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in BANDS if c != 'base'}
    singles = [cost[f'L{L}'] for L in range(5)]
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'singles_sum': round(sum(singles), 4),
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_L12_majority': bool(cost['L1'] + cost['L2'] >= 0.5 * max(cost['front'], 1e-6)),
           'pred_b_superadditive': bool(sum(singles) <= cost['front']),
           'pred_c_L0_free': bool(cost['L0'] <= 0.003),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | singles sum {out['singles_sum']} vs joint {cost['front']}")
    print(f"sanity {out['sanity']} | pred_a L1+L2 {out['pred_a_L12_majority']} | pred_b superadd {out['pred_b_superadditive']} | pred_c L0 {out['pred_c_L0_free']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
