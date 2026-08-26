# mlp0_attn1_edge_rank: THE ATTENTION ANALOG OF THE EDGE-RANK RESULT (user directive:
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
#   pred_a the mlp0 -> attn1-pattern edge is real: cut costs >= .03 CE.
#   pred_b rank-32 recovers >= .70 of the cut -> clean gap.
#   pred_c rank-8 recovers >= .40.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_attn1_edge_rank_results.json'
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


@torch.no_grad()
def fwd_arm(idx, deltas):
    """deltas: dict name->[1152,4608] delta matrix subtracted from the direct
    mlp0 path into block-1's c_q/c_k/c_q2/c_k2 preacts; None -> clean."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; h0 = None
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L == 1 and deltas is not None:
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / xm.float().norm(dim=-1, keepdim=True)
            corr = lam0 * g
            pre = {}
            for nm, mod in (('q', at.c_q), ('k', at.c_k),
                            ('q2', at.c_q2), ('k2', at.c_k2)):
                p = mod(xin).float()
                p = p - corr * (h0 @ deltas[nm].T)
                pre[nm] = p.view(B, T, 9, 128)
            cos, sin = at.rotary(pre['q'])
            q = are(F.rms_norm(pre['q'], (128,)), cos, sin)
            k = are(F.rms_norm(pre['k'], (128,)), cos, sin)
            q2 = are(F.rms_norm(pre['q2'], (128,)), cos, sin)
            k2 = are(F.rms_norm(pre['k2'], (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            ao = at.c_proj(y.reshape(B, T, D))
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
    COMP = {'q': at1.c_q.weight.float().to(DEV) @ Wd0,
            'k': at1.c_k.weight.float().to(DEV) @ Wd0,
            'q2': at1.c_q2.weight.float().to(DEV) @ Wd0,
            'k2': at1.c_k2.weight.float().to(DEV) @ Wd0}

    FR = cl.fineweb_rows(960, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
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
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    rms = (a2 / n0).clamp_min(1e-12).sqrt()
    print("rms(h0) measured", flush=True)

    def rank_deltas(r):
        out = {}
        for nm, M in COMP.items():
            if r == 0:
                out[nm] = M.clone()
            else:
                Mw = M * rms.unsqueeze(0)
                U, S, Vt = torch.linalg.svd(Mw, full_matrices=False)
                Mr = (U[:, :r] * S[:r]) @ Vt[:r]
                out[nm] = M - Mr / rms.unsqueeze(0)
        return out

    def ce_run(deltas):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, deltas).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {'clean': round(ce_run(None), 4)}
    res['cut'] = round(ce_run(rank_deltas(0)), 4)
    print(f"clean {res['clean']} cut {res['cut']}", flush=True)
    for r in (8, 32, 128):
        res[f'rank{r}'] = round(ce_run(rank_deltas(r)), 4)
        print(f"rank{r}: {res[f'rank{r}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    gap = res['cut'] - res['clean']
    rec = lambda a: (res['cut'] - res[a]) / max(gap, 1e-6)
    recs = {f'rank{r}': round(rec(f'rank{r}'), 4) for r in (8, 32, 128)}
    pa = gap >= 0.03
    pb = rec('rank32') >= 0.70
    pc = rec('rank8') >= 0.40
    out = {'ce': res, 'edge_gap_cut': round(gap, 4), 'recovery': recs,
           'pred_a_edge_ge_03': bool(pa), 'pred_b_rank32_70': bool(pb),
           'pred_c_rank8_40': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gap {gap:.4f} recs {recs}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
