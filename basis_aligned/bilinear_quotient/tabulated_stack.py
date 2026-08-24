# tabulated_stack: first COMPOSED reduction — folded patterns + sink-constant, jointly.
#
# §1166: all-162-pattern window folding costs +0.0141. §1089 (sink arc): head 5.7 replaced
# by one constant vector costs 0.013. Both are certified alone; §1168 showed fold errors
# compound 4.1x across layers — do independent REDUCTIONS also interact, or do they compose?
# This is the first step toward the maximally-tabulated model (benchmark north star, causally).
#
# Conditions: base (sanity), foldpat (replicate §1166), sink_const (5.7's output replaced by
# its global mean vector, captured from the base runs of these rows), joint (both).
# Registered predictions:
#   pred_a SINK REPLICATES: sink_const cost <= 0.03 under this harness (§1089: 0.013).
#   pred_b COMPATIBLE: joint <= foldpat + sink_const + 0.02 (no destructive interaction).
#   pred_c CHEAP JOINTLY: joint <= 0.06 nats total.
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'tabulated_stack_results.json'
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
def custom_forward(idx, XH, fold_layers, sink_const=None):
    """Patterns folded at fold_layers; optionally head 5.7's output slice replaced by a
    constant vector (sink_const, dim 128, pre-c_proj head slice)."""
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
        if L == 5 and sink_const is not None:
            y = y.clone(); y[:, :, 7, :] = sink_const.to(y.dtype)
        y = y.reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    CONDS = {'base': (set(), False), 'foldpat': (set(range(18)), False),
             'sink_const': (set(), True), 'joint': (set(range(18)), True)}
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; ntok = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        # true-model sanity CE
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        # capture head 5.7's live output (pre-c_proj slice) on these rows for the constant
        capv = {}
        def hk(mo, i_, o_):
            capv['x57'] = None  # placeholder; we recompute below
        # recompute 5.7 output within a capture-forward: reuse custom_forward pieces inline
        xx = F.rms_norm(m.transformer.wte(idx), (D,)); x00 = xx; vv1 = None
        sink_vec = None
        for L, blk in enumerate(m.transformer.h):
            at = blk.attn
            xm = blk.lambdas[0] * xx + blk.lambdas[1] * x00
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(4, T, 9, 128))
            pat = pattern_from(xin, at, cos, sin)
            v = at.c_v(xin).view(4, T, 9, 128)
            if vv1 is None: vv1 = v
            vmix = (1 - at.lamb) * v + at.lamb * vv1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vmix.dtype), vmix)
            if L == 5:
                sink_vec = y[:, :, 7, :].reshape(-1, 128).float().mean(0)
            xx = xm + at.c_proj(y.reshape(4, T, D))
            xx = xx + blk.mlp(F.rms_norm(xx, (D,)))
        XH = {L: window_resid(idx, W, L) for L in range(18)}
        for cname, (layers, use_sink) in CONDS.items():
            lo = custom_forward(idx, XH, layers, sink_const=sink_vec if use_sink else None)
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in ('foldpat', 'sink_const', 'joint')}
    out = {'n_rows': NR, 'W': W, 'ce': CE, 'cost_vs_base': cost,
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_sink_replicates': bool(cost['sink_const'] <= 0.03),
           'pred_b_compatible': bool(cost['joint'] <= cost['foldpat'] + cost['sink_const'] + 0.02),
           'pred_c_cheap_jointly': bool(cost['joint'] <= 0.06),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a sink {out['pred_a_sink_replicates']} | "
          f"pred_b compatible {out['pred_b_compatible']} | pred_c cheap {out['pred_c_cheap_jointly']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
