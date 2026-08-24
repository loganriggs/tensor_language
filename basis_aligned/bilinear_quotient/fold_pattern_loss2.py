# fold_pattern_loss2: v1's SANITY CONTROL CAUGHT A FORWARD BUG (base CE 10.33 vs true 3.36):
# the block's lambda-mix REPLACES the residual (x = l0*x + l1*x0; x = x + attn_out), but v1's
# custom forward added attention output to the PRE-mix x, discarding the mix from the stream.
# Fixed here (x = xm + c_proj(y)). All v1 condition numbers were void; predictions unchanged.
#
# fold_pattern_loss: THE CAUSAL CAPSTONE of the fold arc — run the model with ALL attention
# patterns computed from weights over the 128-token window variable, and pay the price in nats.
#
# §1162-64 + map2: every head's pattern argmax is 0.68-1.0 window-predictable (162-head map,
# exact-prefix fix; sink 0.998). But this model's attention is UNNORMALIZED (pattern = raw
# (q·k/D)(q2·k2/D), no softmax — §1087's pooled-mass ramp), so the causal bar is higher than
# argmax: the pattern's VALUES multiply v directly. This experiment replaces the pattern at
# every layer with the window-folded one (q/k/q2/k2 from x_hat; v, value-residual, c_proj,
# MLPs all LIVE) and measures CE. Exactness upgrade over the map: the folded side now applies
# the block's λ-mix with the true x0 (a unigram variable) and rms_norm before projections —
# matching the real attention input transform (the map fed raw window residuals).
#
# Conditions (NR=16 rows, W=128, exact-prefix window forward):
#   base           — custom forward, everything live (sanity: must match true-model CE ±0.02)
#   fold_all       — patterns folded at all 18 layers
#   fold_front     — folded at L0-4 only
#   fold_deep      — folded at L5-17 only
#   fold_shuffled  — all folded, x_hat from SHUFFLED rows (wrong text, right machinery) = null
#
# Registered predictions:
#   pred_a SELECTION IS WINDOW-COMPUTABLE CAUSALLY: fold_all costs <= +0.30 nats over base.
#   pred_b NULL CATASTROPHIC: fold_shuffled >= +1.5 nats.
#   pred_c ROUGHLY ADDITIVE: |cost_all − (cost_front + cost_deep)| <= 0.3 × cost_all.
# Sanity: base CE within 0.02 of the standard model forward's CE on the same rows.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_pattern_loss2_results.json'
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
    CONDS = {'base': None, 'fold_all': set(range(18)), 'fold_front': set(range(5)),
             'fold_deep': set(range(5, 18)), 'fold_shuffled': set(range(18))}
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; ntok = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        # true-model sanity CE
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        sidx = idx[torch.randperm(4, device=DEV)]
        XH = {L: window_resid(idx, W, L) for L in range(18)}
        XHs = {L: window_resid(sidx, W, L) for L in range(18)}
        for cname, layers in CONDS.items():
            xh = XHs if cname == 'fold_shuffled' else XH
            lo = custom_forward(idx, xh, layers if layers is not None else set())
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in ('fold_all', 'fold_front', 'fold_deep', 'fold_shuffled')}
    out = {'n_rows': NR, 'W': W, 'ce': CE, 'cost_vs_base': cost,
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_causally_foldable': bool(cost['fold_all'] <= 0.30),
           'pred_b_null_catastrophic': bool(cost['fold_shuffled'] >= 1.5),
           'pred_c_additive': bool(abs(cost['fold_all'] - (cost['fold_front'] + cost['fold_deep']))
                                   <= 0.3 * max(cost['fold_all'], 1e-6)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a foldable {out['pred_a_causally_foldable']} | "
          f"pred_b null {out['pred_b_null_catastrophic']} | pred_c additive {out['pred_c_additive']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
