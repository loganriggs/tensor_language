# repeat_heads2: head grain at the two NEW reader stations L3 and L8 (repeat_range_singles
# found L3 1.01 the TOP single distance reader and L8 0.62 the mid1 anchor — neither was in
# the §1205 head run, but BOTH layers' heads were on §954's identifiable list: L3h8, L8h3/h4).
#
# Same instrument (W=64 read-mask, pos-0 visible, verbatim-repeat rows, scored t>=128),
# per-head masks. Conditions: base; L3; L3H8; L3noH8; L8; L8H3; L8H4; L8H34; quad
# (L2H5+L3H8+L8H3+L8H4 — the four dossier heads now implicated); all18.
#
# Registered predictions:
#   pred_a L3H8 IS L3's READER: cost(L3H8) >= 0.5 x cost(L3) and cost(L3noH8) <= 0.5 x cost(L3).
#   pred_b L8's PAIR CARRIES IT: cost(L8H34) >= 0.6 x cost(L8).
#   pred_c QUAD IS HALF THE CIRCUIT: cost(quad) >= 0.4 x cost(all18) — if TRUE the copy
#          circuit's long-range reads concentrate in 4 nameable heads (vs §1205's pair at 26%);
#          if FALSE, §649's non-localizability keeps winning.
# Controls: sanity base = true model (±0.005); L3/L8 whole-layer must replicate
# repeat_range_singles (±0.05).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_heads2_results.json'
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
             'L3': {3: ALLH}, 'L3H8': {3: {8}}, 'L3noH8': {3: ALLH - {8}},
             'L8': {8: ALLH}, 'L8H3': {8: {3}}, 'L8H4': {8: {4}}, 'L8H34': {8: {3, 4}},
             'quad': {2: {5}, 3: {8}, 8: {3, 4}},
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
           'replicates_singles': bool(abs(cost['L3'] - 1.0076) <= 0.05 and
                                      abs(cost['L8'] - 0.6194) <= 0.05),
           'pred_a_L3H8': bool(cost['L3H8'] >= 0.5 * cost['L3'] and
                               cost['L3noH8'] <= 0.5 * cost['L3']),
           'pred_b_L8_pair': bool(cost['L8H34'] >= 0.6 * cost['L8']),
           'pred_c_quad_half': bool(cost['quad'] >= 0.4 * cost['all18']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | repl {out['replicates_singles']} | pred_a {out['pred_a_L3H8']} | pred_b {out['pred_b_L8_pair']} | pred_c {out['pred_c_quad_half']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
