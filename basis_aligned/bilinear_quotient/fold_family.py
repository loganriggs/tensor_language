# fold_family: is the fold capstone FAMILY-UNIVERSAL? (§1166 on swiglu18.)
#
# bilin18: replacing all 162 attention patterns with weights-only 128-token window functions
# costs +0.0141 nats (§1166). swiglu18 is the independently-trained sibling with STANDARD
# SOFTMAX attention (single q/k, sdpa) — a different selection nonlinearity entirely. If its
# selection is also a bounded-window weights function, the law is a property of the training
# task/family, not of bilin18's unnormalized squared attention.
#
# Method: custom forward mirroring the block scaffold (verified identical: lambdas mix, attn
# on rms_norm, value-residual v1, then MLP); patterns = causal softmax(q̂·k̂/√128) with
# qk-rms-norm + rotary, computed from live x or from window-folded x_hat (W=128, exact-prefix
# fix). Values/c_proj/MLPs live. CE on FineWeb rows. Also a compact argmax map at sampled
# layers {0,1,5,9,13,17} for cross-model comparison with §1165's map.
#
# Conditions: base (sanity vs true forward ±0.02), fold_all (18 layers), fold_shuffled (null).
#
# Registered predictions:
#   pred_a FAMILY-UNIVERSAL: fold_all costs <= 0.30 nats over base (bilin18: 0.0141 — softmax
#          sharpening may amplify score errors, so the bar is the registered 0.30, with the
#          bilin18-magnitude outcome reported either way).
#   pred_b NULL: fold_shuffled >= 1.0 nats.
#   pred_c MAP TRANSFERS: sampled-layer mean argmax hit @W=128 >= 0.6 at every sampled layer.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_family_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 16; W = 128
MAP_LAYERS = [0, 1, 5, 9, 13, 17]
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb


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
        x = F.rms_norm(mdl.transformer.wte(wb), (D,)); x0 = x; v1 = None
        for blk in mdl.transformer.h[:nblocks]:
            x, v1 = blk(x, v1, x0)
        outs.append(x[:, -1].detach())
    res = torch.cat(outs, 0).reshape(B, Tn, D)
    Wp = min(W, Tn)
    xp = F.rms_norm(mdl.transformer.wte(tokens[:, :Wp]), (D,)); x0p = xp; v1p = None
    for blk in mdl.transformer.h[:nblocks]:
        xp, v1p = blk(xp, v1p, x0p)
    res[:, :Wp] = xp.detach()
    return res


def softmax_pattern(xin, at, cos, sin):
    B = xin.shape[0]
    q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
    k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
    q = are(q, cos, sin); k = are(k, cos, sin)
    scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float('-inf'))
    return F.softmax(scores, dim=-1)


@torch.no_grad()
def custom_forward(idx, XH, fold_layers):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        src = F.rms_norm(blk.lambdas[0] * XH[L] + blk.lambdas[1] * x0, (D,)) if L in fold_layers else xin
        pat = softmax_pattern(src, at, cos, sin)
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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    ce = {'base': 0.0, 'fold_all': 0.0, 'fold_shuffled': 0.0}; ce_true = 0.0; ntok = 0
    maphit = {L: [0, 0] for L in MAP_LAYERS}
    ALL = set(range(18))
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
        capd = {}
        for L, blk in enumerate(mdl.transformer.h):
            if L in MAP_LAYERS:
                capd[L] = F.rms_norm(blk.lambdas[0] * x + blk.lambdas[1] * x0, (D,))
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        sidx = idx[torch.randperm(4, device=DEV)]
        XH = {L: window_resid(idx, W, L) for L in range(18)}
        XHs = {L: window_resid(sidx, W, L) for L in range(18)}
        for cname, xh, layers in (('base', XH, set()), ('fold_all', XH, ALL), ('fold_shuffled', XHs, ALL)):
            lo = custom_forward(idx, xh, layers)
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        # compact argmax map at sampled layers
        for L in MAP_LAYERS:
            at = mdl.transformer.h[L].attn; blk = mdl.transformer.h[L]
            cos, sin = at.rotary(at.c_q(capd[L]).view(4, T, 9, 128))
            rp = softmax_pattern(capd[L], at, cos, sin)
            fp = softmax_pattern(F.rms_norm(blk.lambdas[0] * XH[L] + blk.lambdas[1] * x0, (D,)), at, cos, sin)
            for q in range(128, T, 5):
                kr = rp[:, :, q, :q].argmax(-1); kf = fp[:, :, q, :q].argmax(-1)
                maphit[L][0] += int((kr == kf).sum()); maphit[L][1] += kr.numel()
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in ('fold_all', 'fold_shuffled')}
    mh = {str(L): round(h / max(n, 1), 3) for L, (h, n) in maphit.items()}
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': W, 'ce': CE, 'cost_vs_base': cost,
           'map_hit_by_layer': mh, 'bilin18_refs': {'fold_all': 0.0141, 'null': 1.5134},
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_family_universal': bool(cost['fold_all'] <= 0.30),
           'pred_b_null': bool(cost['fold_shuffled'] >= 1.0),
           'pred_c_map_transfers': bool(all(float(v) >= 0.6 for v in mh.values())),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | map hits {mh}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a universal {out['pred_a_family_universal']} | "
          f"pred_b null {out['pred_b_null']} | pred_c map {out['pred_c_map_transfers']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
