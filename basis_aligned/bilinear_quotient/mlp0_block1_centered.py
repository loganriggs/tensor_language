# mlp0_block1_centered: MEAN-PRESERVING EDGE CUTS (S1466: the values edge cut costs
# 1.149 CE — MORE than mlp0's whole delta_opt .908 — because cut-to-zero removes the
# MEAN transport too, which the optimal constant would keep. Here each edge is cut
# CENTERED: subtract corr * ((h0 - mu0) @ C.T) — the mean flows, the data-dependent
# signal is removed. This sizes the true per-edge information content. mu0 from 960
# rows. Arms: centered cuts of the values edge, the pattern edge (all four QK maps),
# and both applied together (block-1 attn reads only mlp0's mean).
# Original header: COMPLETE BLOCK-1'S READ OF mlp0 (S1464: the PATTERN edge is
# .655 CE and rank-8 recovers 98%; mlp1 edge .221 and rank-32 .81). Remaining pathway:
# the VALUES edge Av = c_v1@Down0 — linear, no double-QK. Same exact harness (delta
# subtracted from c_v(xin) before head split; patterns live).
# Original header: THE ATTENTION ANALOG OF THE EDGE-RANK RESULT (user directive:
# "the same applies for the bilinear attn, to a larger degree for the double QK").
# Composed matrices Aq = c_q1@Down0, Ak = c_k1@Down0, Aq2 = c_q2_1@Down0,
# Ak2 = c_k2_1@Down0 (each [1152, 4608]): how block-1's PATTERN computation reads
# mlp0's hidden units. The pattern is (q.k)(q2.k2) — a product of two bilinear scores,
# so an mlp0 direction must register in BOTH factors to move it. S1463 showed rank
# beats sparse at the mlp1 edge, so the arms here are rank truncations (h0-whitened)
# of ALL FOUR matrices simultaneously; values path (c_v) stays live/exact — this
# isolates the mlp0 -> attn1-PATTERN edge. Exactness: delta subtracted from c_*(xin)
# BEFORE the per-head rms_norm and rotary, so the ledger stays exact by construction
# (k=cut is the edge size; no kfull arm needed since delta=0 is identically clean).
#
# Registered predictions:
#   pred_a centered values cut <= .40 CE (>= 65% of the 1.149 was mean transport).
#   pred_b centered pattern cut <= .30 CE (same logic on the .655).
#   pred_c joint centered cut <= 1.2x the sum of the two singles (near-additive).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_block1_centered_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def block_attn(blk, xin, B, v1):
    at = blk.attn
    cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
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
    y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
    return at.c_proj(y.reshape(B, T, D)), v1


MU0 = {'mu': None}


@torch.no_grad()
def fwd_arm(idx, mode, CV, CP):
    """mode None=clean; 'v'/'pat'/'both' = centered cut of that edge at block 1.
    CV: values composition [1152,4608]; CP: dict of 4 pattern compositions."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; h0 = None
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L == 1 and mode is not None:
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / xm.float().norm(dim=-1, keepdim=True)
            corr = lam0 * g
            hc = h0 - MU0['mu']
            if mode in ('pat', 'both'):
                pre = {}
                for nm, mod_ in (('q', at.c_q), ('k', at.c_k),
                                 ('q2', at.c_q2), ('k2', at.c_k2)):
                    p = mod_(xin).float() - corr * (hc @ CP[nm].T)
                    pre[nm] = p.view(B, T, 9, 128)
            else:
                pre = {'q': at.c_q(xin).view(B, T, 9, 128).float(),
                       'k': at.c_k(xin).view(B, T, 9, 128).float(),
                       'q2': at.c_q2(xin).view(B, T, 9, 128).float(),
                       'k2': at.c_k2(xin).view(B, T, 9, 128).float()}
            cos, sin = at.rotary(pre['q'])
            q = are(F.rms_norm(pre['q'], (128,)), cos, sin)
            k = are(F.rms_norm(pre['k'], (128,)), cos, sin)
            q2 = are(F.rms_norm(pre['q2'], (128,)), cos, sin)
            k2 = are(F.rms_norm(pre['k2'], (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            if mode in ('v', 'both'):
                vpre = at.c_v(xin).float() - corr * (hc @ CV.T)
            else:
                vpre = at.c_v(xin).float()
            v = vpre.view(B, T, 9, 128)
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v).float()
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            ao = at.c_proj(y.reshape(B, T, D).to(at.c_proj.weight.dtype))
        else:
            ao, v1n = block_attn(blk, xin, B, v1)
            if v1 is None:
                v1 = v1n
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0:
            h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
        x = x + blk.mlp(z)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    at1 = H[1].attn
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    CV = at1.c_v.weight.float().to(DEV) @ Wd0
    CP = {'q': at1.c_q.weight.float().to(DEV) @ Wd0,
          'k': at1.c_k.weight.float().to(DEV) @ Wd0,
          'q2': at1.c_q2.weight.float().to(DEV) @ Wd0,
          'k2': at1.c_k2.weight.float().to(DEV) @ Wd0}

    FR = cl.fineweb_rows(960, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 960, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, _ = block_attn(blk, xin, idx.shape[0], None)
        xx = xm + ao
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        a1 += h.sum(0); n0 += h.shape[0]
    MU0['mu'] = a1 / n0
    print("mu(h0) measured", flush=True)

    def ce_run(mode):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, mode, CV, CP).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {}
    for mode in (None, 'v', 'pat', 'both'):
        res[str(mode)] = round(ce_run(mode), 4)
        print(f"{mode}: {res[str(mode)]}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    dv = res['v'] - res['None']; dp = res['pat'] - res['None']
    db = res['both'] - res['None']
    pa = dv <= 0.40
    pb = dp <= 0.30
    pc = db <= 1.2 * (dv + dp)
    out = {'ce': res, 'centered_gaps': {'v': round(dv, 4), 'pat': round(dp, 4),
                                       'both': round(db, 4)},
           'uncentered_ref': {'v': 1.1487, 'pat': 0.6552},
           'pred_a_v_le_40': bool(pa), 'pred_b_pat_le_30': bool(pb),
           'pred_c_near_additive': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"centered v {dv:.4f} pat {dp:.4f} both {db:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
