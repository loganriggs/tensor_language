# tabulated_stack3: stack2's m0 entry upgraded k=2 -> k=8 (§1176: diverse-corpus locality).
# Registered: (a) m0big@k8 = 0.014 +/- 0.008; (b) stack3 <= 0.045; (c) sub-additive still.
#
# tabulated_stack2: THIRD entry in the composed-reduction ledger — mlp0 as the writeup-480
# BIGRAM function (weights-only recomputation: attn0 from the embedding with a k=2 causal
# window, the 12.19x lambda-sum mix, manual bilinear MLP), composed with folded patterns
# (§1166) and the sink constant (§1089/§1171).
#
# The 480 construction is copied VERBATIM from m0_context_window.py (the lambda-mix bug that
# invalidated its first version is already fixed there); the hook is GATED so it never fires
# inside window_resid forwards (the §1174 OOM/shape lesson).
#
# Conditions: base (sanity), stack2 (foldpat+sink; §1171 ref 0.0232), m0big (alone; writeup
# 480 ref +0.004 at k=2), stack3 (all three).
# Registered predictions:
#   pred_a M0 REPLICATES IN-HARNESS: m0big alone <= 0.02.
#   pred_b STILL COMPATIBLE: stack3 <= stack2 + m0big + 0.01.
#   pred_c STACK3 CHEAP: stack3 <= 0.05 nats total.
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'tabulated_stack3_results.json'
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


M0G = {'on': False, 'idx': None}
at0g = m.transformer.h[0].attn
mlp0g = m.transformer.h[0].mlp


def _m0_weights():
    mm = m.transformer.h[0].mlp
    L0 = mm.Left.weight.detach().float()
    R0 = mm.Right.weight.detach().float()
    Dn = mm.Down.weight.detach().float()
    Bb = mm.Down_bias.detach().float()
    return L0, R0, Dn, Bb


def m0_bigram_hook(mo, i_, o_):
    if not M0G['on']:
        return None
    idx = M0G['idx']; B, Tn = idx.shape
    if (o_[0] if isinstance(o_, tuple) else o_).shape[:2] != (B, Tn):
        return None                                   # never replace window-forward calls
    E = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = at0g.rotary(at0g.c_q(E).view(B, Tn, 9, 128))
    def rf(w):
        return are(F.rms_norm(w(E).view(B, Tn, 9, 128), (128,)), cos, sin)
    qf, kf = rf(at0g.c_q), rf(at0g.c_k); q2, k2 = rf(at0g.c_q2), rf(at0g.c_k2)
    sc = torch.einsum('bqhd,bkhd->bhqk', qf.float(), kf.float()) / 128
    sc2 = torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128
    ar = torch.arange(Tn, device=DEV)
    mask = torch.tril(torch.ones(Tn, Tn, device=DEV)) * ((ar[:, None] - ar[None, :]) < 8).float()   # k=8 (§1176)
    pat = (sc * sc2) * mask
    v = at0g.c_v(E).view(B, Tn, 9, 128).float()
    z = torch.einsum('bhqk,bkhd->bhqd', pat, v)       # block 0: v1 = v, mix is identity
    a0 = at0g.c_proj(z.transpose(1, 2).contiguous().view(B, Tn, -1).to(E.dtype)).float()
    lam = m.transformer.h[0].lambdas.detach().float()
    xin = F.rms_norm(float(lam.sum()) * E.float() + a0, (D,))
    L0, R0, Dn, Bb = _m0_weights()
    out = ((xin @ L0.T) * (xin @ R0.T)) @ Dn.T + Bb
    ref = o_[0] if isinstance(o_, tuple) else o_
    return out.to(ref.dtype)


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
    CONDS = {'base': (set(), False, False), 'stack2': (set(range(18)), True, False),
             'm0big': (set(), False, True), 'stack3': (set(range(18)), True, True)}
    hk = m.transformer.h[0].mlp.register_forward_hook(m0_bigram_hook)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; ntok = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        # true-model sanity CE
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        ce_true += float(F.cross_entropy(lt.reshape(-1, lt.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        # recompute 5.7 output within a capture-forward for the sink constant
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
        for cname, (layers, use_sink, use_m0) in CONDS.items():
            M0G['on'] = use_m0; M0G['idx'] = idx
            lo = custom_forward(idx, XH, layers, sink_const=sink_vec if use_sink else None)
            M0G['on'] = False
            ce[cname] += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), tgt.reshape(-1), reduction='sum'))
        ntok += idx.numel()
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / ntok, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / ntok, 4)
    hk.remove()
    cost = {c: round(CE[c] - CE['base'], 4) for c in ('stack2', 'm0big', 'stack3')}
    out = {'n_rows': NR, 'W': W, 'ce': CE, 'cost_vs_base': cost,
           'refs': {'stack2_1171': 0.0232, 'm0big_480': 0.004},
           'sanity_base_matches_true': bool(abs(CE['base'] - CE['true_model']) <= 0.02),
           'pred_a_m0_replicates': bool(abs(cost['m0big'] - 0.014) <= 0.008),
           'pred_b_compatible': bool(cost['stack3'] <= cost['stack2'] + cost['m0big'] + 0.01),
           'pred_c_stack3_cheap': bool(cost['stack3'] <= 0.045),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost}")
    print(f"sanity {out['sanity_base_matches_true']} | pred_a m0 {out['pred_a_m0_replicates']} | "
          f"pred_b compatible {out['pred_b_compatible']} | pred_c cheap {out['pred_c_stack3_cheap']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
