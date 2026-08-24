# fold_cost_by_layer: which layers' selection actually needs window width? (§1167 follow-up)
#
# §1166-67: all-162 folding costs +0.014 @W128, +0.229 @W32. This run folds ONE layer at a
# time at W=32 (where cost is measurable) to attribute the price, plus the joint fold for the
# additivity check. Registered predictions:
#   pred_a SUB/SUPER-ADDITIVITY MEASURED: sum of per-layer costs within [0.5x, 2x] of the
#          joint all_W32 cost (0.2285).
#   pred_b L1 IS THE FRONT'S PRICIEST: layer 1's cost is the max over L0-4 (the map's laggard,
#          §1165: argmax 0.68).
#   pred_c DEEP CARRIES IT: layers 5-17 sum to >= 80% of the total per-layer sum.
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_cost_by_layer_results.json'
NR = 16; W = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def window_resid(tokens, W, nblocks):
    B, Tn = tokens.shape
    idx = torch.arange(Tn, device=DEV)
    win = torch.stack([tokens[:, (idx + o).clamp_min(0)] for o in range(-(W - 1), 1)], -1)
    flat = win.reshape(B * Tn, W)
    outs = []
    step = max(128, 4096 // W)
    for i in range(0, flat.shape[0], step):
        wb = flat[i:i + step]
        x = F.rms_norm(m.transformer.wte(wb), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h[:nblocks]:
            x, v1 = blk(x, v1, x0)
        outs.append(x[:, -1].detach())
    res = torch.cat(outs, 0).reshape(B, Tn, D)
    Wp = min(W, Tn)
    xp = F.rms_norm(m.transformer.wte(tokens[:, :Wp]), (D,)); x0p = xp; v1p = None
    for blk in m.transformer.h[:nblocks]:
        xp, v1p = blk(xp, v1p, x0p)
    res[:, :Wp] = xp.detach()
    return res


def pattern_from(xin, at, cos, sin):
    B = xin.shape[0]
    q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
    k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
    q2 = F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,))
    k2 = F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,))
    q = are(q, cos, sin); k = are(k, cos, sin); q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat.masked_fill(~mask, 0.0)


@torch.no_grad()
def custom_forward(idx, XH, fold_layers):
    """Forward with patterns folded (from XH residuals) at fold_layers; all else live."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        if L in fold_layers:
            xm_h = blk.lambdas[0] * XH[L] + blk.lambdas[1] * x0
            pat = pattern_from(F.rms_norm(xm_h, (D,)), at, cos, sin)
        else:
            pat = pattern_from(xin, at, cos, sin)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        y = y.reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    CONDS = {'base': None, 'all_W32': set(range(18))}
    for L in range(18):
        CONDS[f'L{L}'] = {L}
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; ntok = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        # true-model sanity CE
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        XH32 = {L: window_resid(idx, 32, L) for L in range(18)}
        for cname, layers in CONDS.items():
            lo = custom_forward(idx, XH32, layers if layers is not None else set())
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    per = [cost[f'L{L}'] for L in range(18)]
    tot = sum(per); deep = sum(per[5:])
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'per_layer_sum': round(tot, 4), 'joint_all_W32': cost['all_W32'],
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_additivity': bool(0.5 * cost['all_W32'] <= tot <= 2 * cost['all_W32']),
           'pred_b_L1_front_max': bool(cost['L1'] == max(cost[f'L{L}'] for L in range(5))),
           'pred_c_deep_carries': bool(deep >= 0.8 * max(tot, 1e-6)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"per-layer costs {[round(v,4) for v in per]}")
    print(f"sum {round(tot,4)} vs joint {cost['all_W32']}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a additivity {out['pred_a_additivity']} | "
          f"pred_b L1 front-max {out['pred_b_L1_front_max']} | pred_c deep {out['pred_c_deep_carries']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
