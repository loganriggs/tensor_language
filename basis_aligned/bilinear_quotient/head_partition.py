# head_partition: USER PROPOSAL — decompose each head into "the part that does induction"
# and "the rest", instead of treating whole heads as circuit members. The architecture
# hands us the first cut for free: every head's delivered value is
#   v_mixed = (1-lambda)*v_fresh + lambda*v1     (v1 = block-0 identity code, broadcast)
# and §1236 proved the v1 route is the copying substrate GLOBALLY. New question: is it the
# induction part PER HEAD — for the four main heads AND for the long tail — such that
# masking only that route removes the induction contribution while sparing each head's
# other functions?
#
# Targets: natural induction positions — next token continues an earlier occurrence of the
# current token (toks[q]==toks[p], tgt[q]==tgt[p], q in [p-128, p-1]). Conditions (route
# masks applied at ALL positions, zeroing the route in the value mix):
#   base | main4 {2.5, 3.8, 8.3, 8.4}: whole / v1-only / fresh-only
#   tail (all L1-17 heads EXCEPT main4): whole / v1-only
#   all L1-17 heads: v1-only / fresh-only
#
# Registered predictions:
#   pred_a ROUTE = TASK PART (main4): v1-route masking reproduces >= 80% of whole-head
#          induction damage while costing <= 40% of whole-head elsewhere damage (the four
#          heads' other functions live in their fresh route).
#   pred_b TAIL TOO: tail v1-route masking captures >= 60% of tail whole-head induction
#          damage with <= 30% of its elsewhere damage.
#   pred_c v1-SPECIFICITY: the induction/elsewhere damage ratio of the all-heads v1 mask
#          is >= 5x that of the all-heads fresh mask (the v1 route is disproportionately
#          the induction part; the fresh route is disproportionately everything else).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_partition_results.json'
NR = 384; W = 128
H = m.transformer.h
MAIN4 = {(2, 5), (3, 8), (8, 3), (8, 4)}


def selmask(layer, cond):
    """(9,) bool of heads selected at this layer under cond=(set_name, spec)."""
    name = cond
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
def fwd(idx, which=None, route=None):
    """route: 'whole' (zero head), 'v1' (keep fresh only), 'fresh' (keep v1 only)."""
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
        v1v = v1.view_as(v)
        vv = (1 - at.lamb) * v + at.lamb * v1v
        sel = selmask(L, which)
        if sel.any():
            sm = sel.to(DEV).view(1, 1, 9, 1)
            if route == 'whole':
                vv = torch.where(sm, torch.zeros_like(vv), vv)
            elif route == 'v1':
                vv = torch.where(sm, (1 - at.lamb) * v, vv)
            elif route == 'fresh':
                vv = torch.where(sm, at.lamb * v1v, vv)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt = ROWS[:, 1:]
    # natural induction targets: some q in [p-128, p-1] with toks[q]==toks[p], tgt[q]==tgt[p]
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        TGT[b0:b0 + 64] = (eq & band).any(1)
    TGT[:, :16] = False
    ELSE = ~TGT; ELSE[:, :16] = False
    print(f"induction targets {int(TGT.sum())} / {TGT.numel()}", flush=True)

    def ce_sets(which, route):
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx, which, route).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None, None)
    print(f"base {base}", flush=True)
    CONDS = [('main4', 'whole'), ('main4', 'v1'), ('main4', 'fresh'),
             ('tail', 'whole'), ('tail', 'v1'),
             ('all', 'v1'), ('all', 'fresh')]
    res = {}
    for which, route in CONDS:
        r = ce_sets(which, route)
        res[f'{which}_{route}'] = {'d_ind': round(r['t'] - base['t'], 4),
                                   'd_else': round(r['e'] - base['e'], 4)}
        print(f"{which}/{route}: ind {res[f'{which}_{route}']['d_ind']} else {res[f'{which}_{route}']['d_else']}", flush=True)

    m4w, m4v = res['main4_whole'], res['main4_v1']
    tw, tv = res['tail_whole'], res['tail_v1']
    av, af = res['all_v1'], res['all_fresh']
    pa = (m4v['d_ind'] >= 0.8 * m4w['d_ind'] and
          m4v['d_else'] <= 0.4 * max(m4w['d_else'], 1e-4))
    pb = (tv['d_ind'] >= 0.6 * tw['d_ind'] and
          tv['d_else'] <= 0.3 * max(tw['d_else'], 1e-4))
    rat_v = av['d_ind'] / max(av['d_else'], 1e-4)
    rat_f = af['d_ind'] / max(af['d_else'], 1e-4)
    pc = rat_v >= 5 * max(rat_f, 1e-4)
    out = {'n_rows': NR, 'n_targets': int(TGT.sum()),
           'base': {k: round(v, 4) for k, v in base.items()}, 'conds': res,
           'spec_ratio_v1': round(rat_v, 2), 'spec_ratio_fresh': round(rat_f, 2),
           'pred_a_main4_route': bool(pa), 'pred_b_tail_route': bool(pb),
           'pred_c_v1_specific': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a main4 {pa} | pred_b tail {pb} | pred_c specificity {pc} (v1 ratio {rat_v:.2f} vs fresh {rat_f:.2f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
