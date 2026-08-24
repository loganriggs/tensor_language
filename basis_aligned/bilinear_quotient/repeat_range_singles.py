# repeat_range_singles: crowd or chain INSIDE each band? Per-layer read-masks on repeat rows.
#
# repeat_range_bands found copying SERIAL ACROSS bands (front 1.91 and mid1 2.64 each alone
# catastrophic; band sum 4.70 > joint 3.20 — opposite of prose §1187). repeat_range_heads
# found L2 the dominant single layer (0.53, its head H5 0.62) yet far below its band's 1.91,
# and L5 small (0.12). Open question this run answers: is the remainder WITHIN each band a
# redundant crowd (prose-style: singles tiny, band huge) or more chain links (singles large,
# summing toward the band)?
#
# Instrument: identical to repeat_range_bands (W=64 read-mask, pos-0 visible, verbatim-repeat
# rows, scored t>=128). Conditions: base + each single layer L0..L9 + front + mid1 (anchors).
#
# Registered predictions:
#   pred_a L2 TOP SINGLE: cost(L2) >= 3x every other single layer's cost.
#   pred_b FRONT IS A CROWD BEYOND L2: sum of front singles (L0-4) <= 0.6 x cost(front) —
#          the band's 1.91 is mostly joint (redundant backup among front layers).
#   pred_c MID1 IS A CROWD: every mid1 single (L5-9) <= 0.15 and their sum <= 0.5 x
#          cost(mid1) — §1187's prose redundancy pattern holds within-band in the copy
#          regime too, even though ACROSS bands the circuit is serial.
# Controls: sanity base = true model (±0.005); front/mid1 must replicate repeat_range_bands
# (±0.05) — same rows/instrument, drift means a harness bug.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_range_singles_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None


def make_mask():
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


@torch.no_grad()
def forward_banded(idx, band):
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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': set()}
    for L in range(10):
        CONDS[f'L{L}'] = {L}
    CONDS['front'] = set(range(5)); CONDS['mid1'] = set(range(5, 10))
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, band in CONDS.items():
            lo = forward_banded(idx, band).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    fsum = sum(cost[f'L{L}'] for L in range(5))
    msum = sum(cost[f'L{L}'] for L in range(5, 10))
    others = [cost[f'L{L}'] for L in range(10) if L != 2]
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'front_singles_sum': round(fsum, 4), 'mid1_singles_sum': round(msum, 4),
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'replicates_bands': bool(abs(cost['front'] - 1.9124) <= 0.05 and
                                    abs(cost['mid1'] - 2.6438) <= 0.05),
           'pred_a_L2_top': bool(all(cost['L2'] >= 3 * o for o in others)),
           'pred_b_front_crowd': bool(fsum <= 0.6 * cost['front']),
           'pred_c_mid1_crowd': bool(all(cost[f'L{L}'] <= 0.15 for L in range(5, 10)) and
                                     msum <= 0.5 * cost['mid1']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | front singles sum {out['front_singles_sum']} vs band {cost['front']} | mid1 singles sum {out['mid1_singles_sum']} vs band {cost['mid1']}")
    print(f"sanity {out['sanity']} | repl {out['replicates_bands']} | pred_a {out['pred_a_L2_top']} | pred_b {out['pred_b_front_crowd']} | pred_c {out['pred_c_mid1_crowd']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
