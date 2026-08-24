# fold_width_law: the causal window-width law for attention selection (§1166 follow-up).
#
# §1166: all-162-pattern replacement at W=128 costs +0.0141 nats. How short can the window
# get? Same harness, fold_all at W in {16, 32, 64, 128} plus fold_front at W=16.
# Registered predictions:
#   pred_a MONOTONE: fold_all cost falls with W at every step.
#   pred_b W=64 SUFFICES: fold_all cost <= 0.05 nats at W=64.
#   pred_c FRONT IS N-GRAM-SHORT: fold_front cost <= 0.005 at W=16.
# Null and sanity inherited from §1166 (machinery certified there; base re-checked).
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_width_law_results.json'
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
    CONDS = {'base': None, 'all_W16': set(range(18)), 'all_W32': set(range(18)),
             'all_W64': set(range(18)), 'all_W128': set(range(18)), 'front_W16': set(range(5))}
    CONDW = {'all_W16': 16, 'all_W32': 32, 'all_W64': 64, 'all_W128': 128, 'front_W16': 16}
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; ntok = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        # true-model sanity CE
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        XHW = {w: {L: window_resid(idx, w, L) for L in range(18)} for w in (16, 32, 64, 128)}
        for cname, layers in CONDS.items():
            xh = XHW[CONDW[cname]] if cname != 'base' else XHW[16]
            lo = custom_forward(idx, xh, layers if layers is not None else set())
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    seq = [cost['all_W16'], cost['all_W32'], cost['all_W64'], cost['all_W128']]
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_monotone': bool(all(seq[j+1] < seq[j] for j in range(3))),
           'pred_b_w64_suffices': bool(cost['all_W64'] <= 0.05),
           'pred_c_front_short': bool(cost['front_W16'] <= 0.005),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a monotone {out['pred_a_monotone']} | "
          f"pred_b W64 {out['pred_b_w64_suffices']} | pred_c front {out['pred_c_front_short']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
