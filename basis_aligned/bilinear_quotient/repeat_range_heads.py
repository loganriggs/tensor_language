# repeat_range_heads: do the DOSSIER-NAMED induction heads carry the copy regime's
# long-range reading, or is it collective like prose pooling? (repeat_range_bands sibling.)
#
# Same instrument and rows as repeat_range_bands (verbatim-repeat rows, read-mask W=64,
# pos-0 visible, scored t>=128) but the mask is applied PER HEAD: only the named head(s)
# lose long-range reads; every other head at that layer — and every other layer — stays
# full-context. modules/induction.md names L2H5 (top single-head score +0.123) and L5H5
# (the gate; pattern share 0.248) as THE identifiable induction heads, yet §649/§952-953
# found copying collectively carried. This is the direct causal test at the regime where
# copying actually binds.
#
# Conditions: base; L2 (whole layer); L2H5 (one head); L5; L5H5; L5noH5 (all L5 heads
# EXCEPT H5); L2H5+L5H5 (both named heads jointly); all18 (scale anchor).
#
# Registered predictions:
#   pred_a NAMED HEADS ARE THE LAYERS: cost(L2H5) >= 0.5 x cost(L2) and
#          cost(L5H5) >= 0.5 x cost(L5) — within each layer the induction head is the
#          long-range reader (complement check: cost(L5noH5) <= 0.5 x cost(L5)).
#   pred_b BUT THE PAIR IS NOT THE CIRCUIT: cost(L2H5+L5H5) <= 0.3 x cost(all18) —
#          redundancy across the rest of the network backs them up (§952-953).
#   pred_c LAYER ORDER: cost(L5) >= cost(L2) — the gate outranks the front reader on
#          repeat text (ablating L5 collapses induction, §877-878).
# Controls: base = unmasked custom forward vs true model (sanity ±0.005); repeat rows
# easy (base <= 0.5 nats).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_range_heads_results.json'
NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None
ALLH = set(range(9))


def make_mask():
    ar = torch.arange(T, device=DEV)
    near = (ar[:, None] - ar[None, :]) < WIN
    vis = near | (ar[None, :] == 0)
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


@torch.no_grad()
def forward_headmasked(idx, spec):
    """spec: dict layer -> set of head indices whose pattern is window-masked."""
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
    CONDS = {'base': {},
             'L2': {2: ALLH}, 'L2H5': {2: {5}},
             'L5': {5: ALLH}, 'L5H5': {5: {5}}, 'L5noH5': {5: ALLH - {5}},
             'pair': {2: {5}, 5: {5}},
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
           'repeat_easy': bool(CE['base'] <= 0.5),
           'pred_a_heads_are_layers': bool(cost['L2H5'] >= 0.5 * cost['L2'] and
                                           cost['L5H5'] >= 0.5 * cost['L5'] and
                                           cost['L5noH5'] <= 0.5 * cost['L5']),
           'pred_b_pair_not_circuit': bool(cost['pair'] <= 0.3 * cost['all18']),
           'pred_c_gate_outranks': bool(cost['L5'] >= cost['L2']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | easy {out['repeat_easy']} | pred_a {out['pred_a_heads_are_layers']} | pred_b {out['pred_b_pair_not_circuit']} | pred_c {out['pred_c_gate_outranks']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
