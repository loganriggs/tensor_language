# repeat_range_bands: WHO carries the copying regime's long-range reading? (§1195 follow-on.)
#
# §1195: on verbatim-repeat rows (tokens[128:256]=tokens[0:128]) the model nearly solves the
# second half (base CE 0.258) and read-windows cost 2.7-3.4 nats — 30x the natural-prose
# budget. That cost was measured whole-model only. This maps it at BAND grain with the exact
# §1186 instrument: window-restrict attention READS (pattern masked beyond W=64, position 0
# always visible) one depth band at a time, on repeat rows. At scored positions t>=128 the
# copy source t-128 is outside every window — masking a band severs THAT band's copy-reading
# and nothing else (MLPs, values within window, other bands full-context).
#
# Conditions: base; front (L0-4); mid1 (L5-9); mid2 (L10-14); late (L15-17); all18.
# Scored positions >= 128, stride 1. Rows: FineWeb, second half verbatim repeat of first.
#
# Dossier stakes (modules/induction.md): the induction chain is FRONT-loaded — attn0 writes
# copy-source, L2h5 is the top-scoring head, L5h5 gates (collapse when L5 ablated) — but
# copying was also shown NON-localizable (§649) and collectively carried (§952-953). Prose
# pooling lives in mid1 as a redundant crowd (§1186-87). Which geography does COPYING have?
#
# Registered predictions:
#   pred_a FRONT+MID1 CARRY IT: cost(front) + cost(mid1) >= 0.7 x cost(all18) — the
#          induction chain (L2 reader + L5 gate) sits in those two bands.
#   pred_b LATE IS LOCAL EVEN HERE: cost(late) <= 0.05 x cost(all18) — readout consumes
#          own-position coordinates in every regime.
#   pred_c REDUNDANT LIKE PROSE: band sum <= 0.8 x cost(all18) (super-additive joint —
#          copying is collectively carried, §649/§952-953; no band is indispensable alone).
# Controls: base = unmasked custom forward, must equal true-model CE (sanity ±0.005);
# base CE must replicate §1195's easy-repeat figure (<= 0.5 nats) or rows are wrong.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_range_bands_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None


def make_mask():
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)                       # position 0 always visible
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]                     # verbatim repeat (§1195 rows)
    BANDS = {'base': set(), 'front': set(range(5)), 'mid1': set(range(5, 10)),
             'mid2': set(range(10, 15)), 'late': {15, 16, 17}, 'all18': set(range(18))}
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
    bandsum = cost['front'] + cost['mid1'] + cost['mid2'] + cost['late']
    out = {'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'band_sum': round(bandsum, 4),
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'repeat_easy': bool(CE['base'] <= 0.5),
           'pred_a_front_mid1': bool(cost['front'] + cost['mid1'] >= 0.7 * cost['all18']),
           'pred_b_late_local': bool(cost['late'] <= 0.05 * cost['all18']),
           'pred_c_redundant': bool(bandsum <= 0.8 * cost['all18']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | band sum {out['band_sum']} vs all18 {cost['all18']}")
    print(f"sanity {out['sanity']} | easy {out['repeat_easy']} | pred_a {out['pred_a_front_mid1']} | pred_b {out['pred_b_late_local']} | pred_c {out['pred_c_redundant']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
