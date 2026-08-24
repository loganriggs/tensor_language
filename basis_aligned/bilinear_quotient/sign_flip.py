# sign_flip: causal test of §1239's sign conventions. Matchers 2.5/3.8 deliver -(matched
# value); fetchers are a sign-opposed differential (8.4 +, 8.3 -). If downstream consumers
# read these conventions (rather than |value|), FLIPPING a head's pattern sign (pat -> -pat,
# that head only, everything else intact) should break copying — and flipping ONE arm of a
# differential pair should hurt MORE than flipping BOTH (which preserves their difference
# up to global sign).
#
# Conditions (repeat rows, scored t>=128): base; flip25; flip38; flip25+38; flip83; flip84;
# flip83+84; flip_all_four. Zero-information control: flipping a near-zero-cost head (17.0,
# screen bottom §1222) must cost ~0.
#
# Registered predictions:
#   pred_a SIGN IS LOAD-BEARING: cost(flip38) >= 0.5 nats (the biggest matcher's convention
#          matters as much as its reads, cf. read-mask 1.08).
#   pred_b DIFFERENTIAL READ: cost(flip83) and cost(flip84) each >= 1.5 x cost(flip83+84)
#          (flipping both preserves the pair difference; flipping one unbalances it).
#   pred_c CONTROL: cost(flip 17.0) <= 0.02; sanity base = true model (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'sign_flip_results.json'
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
        pat = pat.masked_fill(~FULL_TRIL, 0.0)
        heads = spec.get(L, None)
        if heads is not None:
            sv = torch.ones(9, device=pat.device)
            for h in heads:
                sv[h] = -1.0
            pat = pat * sv.view(1, 9, 1, 1)
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
             'flip25': {2: {5}}, 'flip38': {3: {8}}, 'flip2538': {2: {5}, 3: {8}},
             'flip83': {8: {3}}, 'flip84': {8: {4}}, 'flip8384': {8: {3, 4}},
             'flip4': {2: {5}, 3: {8}, 8: {3, 4}}, 'flip170': {17: {0}}}
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
           'pred_a_sign_loadbearing': bool(cost['flip38'] >= 0.5),
           'pred_b_differential': bool(cost['flip83'] >= 1.5 * max(cost['flip8384'], 1e-6) and
                                       cost['flip84'] >= 1.5 * max(cost['flip8384'], 1e-6)),
           'pred_c_control': bool(cost['flip170'] <= 0.02),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a load {out['pred_a_sign_loadbearing']} | pred_b diff {out['pred_b_differential']} | pred_c ctrl {out['pred_c_control']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
