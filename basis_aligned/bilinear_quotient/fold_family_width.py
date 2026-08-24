# fold_family_width: swiglu18 width law — family symmetry for §1167.
# bilin18: fold_all cost 0.482/0.229/0.067/0.014 @ W=16/32/64/128 (accelerating decay).
# Registered: (a) swiglu18 monotone decreasing across W=16/32/64/128;
# (b) W=128 reproduces §1170 (0.0148 ± 0.005); (c) decay factor per doubling GROWS (accelerating,
# matching bilin18's 2.1x -> 3.4x -> 4.8x within a factor of 2 at each step).
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_family_width_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 16; W = 128
MAP_LAYERS = []
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
    ce = {'base': 0.0, 'all_W16': 0.0, 'all_W32': 0.0, 'all_W64': 0.0, 'all_W128': 0.0}
    ce_true = 0.0; ntok = 0
    ALL = set(range(18))
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for L, blk in enumerate(mdl.transformer.h):
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        for w in (16, 32, 64, 128):
            XHw = {L: window_resid(idx, w, L) for L in range(18)}
            lo = custom_forward(idx, XHw, ALL)
            ce[f'all_W{w}'] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        lo = custom_forward(idx, {L: None for L in range(18)}, set())
        ce['base'] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in ('all_W16', 'all_W32', 'all_W64', 'all_W128')}
    seq = [cost['all_W16'], cost['all_W32'], cost['all_W64'], cost['all_W128']]
    ratios = [round(seq[j] / max(seq[j + 1], 1e-6), 2) for j in range(3)]
    bratios = [2.1, 3.4, 4.8]
    out = {'model': 'swiglu18', 'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'decay_ratios_per_doubling': ratios, 'bilin18_ratios': bratios,
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_monotone': bool(all(seq[j + 1] < seq[j] for j in range(3))),
           'pred_b_w128_reproduces': bool(abs(cost['all_W128'] - 0.0148) <= 0.005),
           'pred_c_accelerating_matched': bool(all(0.5 * bratios[j] <= ratios[j] <= 2 * bratios[j] for j in range(3))),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | ratios {ratios} (bilin18 {bratios})")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a monotone {out['pred_a_monotone']} | "
          f"pred_b W128 {out['pred_b_w128_reproduces']} | pred_c ratios {out['pred_c_accelerating_matched']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
