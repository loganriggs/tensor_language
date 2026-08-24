# head_partition2: CUT 2 of the user's head-partition program. §1290 showed the identity
# variable reaches heads by two paths (lambda-broadcast AND stream re-extraction), so the
# partition must be by CONTENT, not route: per head-index h, fit the identity-code
# subspace (top-16 uncentered PCA of block-0's v1[:, :, h, :] over fit rows) and mask the
# projection of the head's FULL mixed value vv onto it — catching the code whichever path
# delivered it. Mandatory null (§1264 lesson): a random orthonormal 16-dim basis per head.
#
# Conditions: base | main4_sub | tail_sub | all_sub | all_rand | main4_rand.
# Registered predictions:
#   pred_a CONTENT IS THE PART (main4): subspace mask reproduces >= 80% of main4's
#          whole-head induction damage (0.395 anchor, §1290) with elsewhere <= 40% of
#          whole-head elsewhere (0.0605).
#   pred_b NULL CLEAN: all_rand induction damage <= 25% of all_sub's.
#   pred_c TAIL IS THE SAME VARIABLE: tail_sub induction damage >= 0.4 nats with
#          elsewhere <= 0.5x its induction damage (the long tail's induction part is the
#          same 16-dim code, and masking it spares the tail's other functions).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_partition2_results.json'
NFIT = 24; NR = 384; W = 128; R = 16
H = m.transformer.h
MAIN4 = {(2, 5), (3, 8), (8, 3), (8, 4)}


def selmask(layer, name):
    s = torch.zeros(9, dtype=torch.bool)
    if name is None or layer == 0:
        return s
    if name == 'main4':
        for (L, h) in MAIN4:
            if L == layer:
                s[h] = True
    elif name == 'tail':
        s[:] = True
        for (L, h) in MAIN4:
            if L == layer:
                s[h] = False
    elif name == 'all':
        s[:] = True
    return s


@torch.no_grad()
def fwd(idx, which=None, bases=None):
    """bases: (9, R, 128) per-head-index orthonormal rows; mask vv's projection onto them
    for selected heads."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        are = sys.modules[type(at).__module__].apply_rotary_emb
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        sel = selmask(L, which)
        if sel.any() and bases is not None:
            proj = torch.einsum('bthd,hrd->bthr', vv.float(), bases)
            comp = torch.einsum('bthr,hrd->bthd', proj, bases).to(vv.dtype)
            sm = sel.to(DEV).view(1, 1, 9, 1)
            vv = torch.where(sm, vv - comp, vv)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)[:, :T + 1].contiguous()
    FITR, EVR = ROWS[:NFIT], ROWS[NFIT:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        TGT[b0:b0 + 64] = (eq & band).any(1)
    TGT[:, :16] = False
    ELSE = ~TGT; ELSE[:, :16] = False
    print(f"induction targets {int(TGT.sum())}", flush=True)

    # fit identity bases: block-0 c_v on fit rows
    caps = []
    hk = H[0].attn.c_v.register_forward_hook(
        lambda mod, args, out: caps.append(out.detach().float().view(out.shape[0], -1, 9, 128)))
    for i in range(0, NFIT, 4):
        fwd(FITR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    v1s = torch.cat(caps, 0)                                   # (NFIT, T, 9, 128)
    bases = torch.zeros(9, R, 128, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(11)
    rand_bases = torch.zeros(9, R, 128, device=DEV)
    for h in range(9):
        M = v1s[:, :, h, :].reshape(-1, 128)
        _, S, Vh = torch.linalg.svd(M, full_matrices=False)
        bases[h] = Vh[:R]
        frac = float((S[:R] ** 2).sum() / (S ** 2).sum())
        rb = torch.randn(128, R, generator=g)
        Qr, _ = torch.linalg.qr(rb)
        rand_bases[h] = Qr.T.to(DEV)
        print(f"head-slice {h}: top-{R} energy {frac:.3f}", flush=True)

    def ce_sets(which, B):
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, which, B).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None, None)
    print(f"base {base}", flush=True)
    res = {}
    for name, which, B in (('main4_sub', 'main4', bases), ('tail_sub', 'tail', bases),
                           ('all_sub', 'all', bases), ('all_rand', 'all', rand_bases),
                           ('main4_rand', 'main4', rand_bases)):
        r = ce_sets(which, B)
        res[name] = {'d_ind': round(r['t'] - base['t'], 4), 'd_else': round(r['e'] - base['e'], 4)}
        print(f"{name}: ind {res[name]['d_ind']} else {res[name]['d_else']}", flush=True)

    pa = (res['main4_sub']['d_ind'] >= 0.8 * 0.395 and
          res['main4_sub']['d_else'] <= 0.4 * 0.0605)
    pb = res['all_rand']['d_ind'] <= 0.25 * max(res['all_sub']['d_ind'], 1e-4)
    pc = (res['tail_sub']['d_ind'] >= 0.4 and
          res['tail_sub']['d_else'] <= 0.5 * max(res['tail_sub']['d_ind'], 1e-4))
    out = {'n_rows': NR, 'n_targets': int(TGT.sum()), 'rank': R,
           'base': {k: round(v, 4) for k, v in base.items()}, 'conds': res,
           'anchors_1290': {'main4_whole': [0.395, 0.0605]},
           'pred_a_content_is_part': bool(pa), 'pred_b_null_clean': bool(pb),
           'pred_c_tail_same_variable': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a content {pa} | pred_b null {pb} | pred_c tail {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
