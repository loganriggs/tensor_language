# edge_atlas_part2: THE FULL ADJACENT-EDGE ATLAS, PART 2 (user directive 2026-08-26:
# "folding in is solid — do more, parallelize; connections and ablations are
# parallelizable, naming stays sequential"). For every source layer L in LRANGE,
# the three direct edges into block L+1 — PATTERN (c_q/c_k/c_q2/c_k2 @ Down_L),
# VALUES (c_v @ Down_L), MLP (Left/Right @ Down_L) — each measured two ways:
#   cut_centered   — mean-preserving cut = the edge's SIGNAL size (S1468 method).
#   beyond_r32     — centered cut of everything beyond the top-32 whitened directions
#                    (is the signal low-rank, as at L=0?).
# ONE stats pass (mu, rms of every mlp hidden, 480 rows); NR=480 screening resolution
# (registered assumption: +-.005; edges > .03 get NR=960 follow-ups). Rank-32 via
# torch.svd_lowrank q=64 (registered assumption: approximate top subspace).
# Batching amortizes: one script = LAYN layers x 3 edges x 2 arms + 1 clean.
#
# Registered predictions (this part):
#   pred_a median centered signal across this part's edges >= .01 CE.
#   pred_b >= 80% of edges with signal >= .005 have beyond_r32 <= .30 x signal
#          (the low-rank law generalizes beyond layer 0).
#   pred_c the MLP edge is the largest of the three at >= 60% of this part's layers.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'edge_atlas_part2_results.json'
NR = 480
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
LRANGE = list(range(8, 17))
STATE = {'mu': {}, 'rms': {}}


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
def mlp_hidden(blk, z):
    return blk.mlp.Left(z).float() * blk.mlp.Right(z).float()


@torch.no_grad()
def fwd_arm(idx, Lsrc, kind, DELTA):
    """DELTA None = clean. Else centered cut at block Lsrc+1 of the given kind:
    'pat' (dict of 4), 'val' (matrix), 'mlp' (dict l/r)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; hsrc = None
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        tgt = (DELTA is not None and L == Lsrc + 1)
        if tgt and kind in ('pat', 'val'):
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / xm.float().norm(dim=-1, keepdim=True)
            corr = lam0 * g
            hc = hsrc - STATE['mu'][Lsrc]
            if kind == 'pat':
                pre = {}
                for nm, mod_ in (('q', at.c_q), ('k', at.c_k),
                                 ('q2', at.c_q2), ('k2', at.c_k2)):
                    p = mod_(xin).float() - corr * (hc @ DELTA[nm].T)
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
            else:
                cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
                q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
                k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
                q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
                k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
                pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                    * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
                tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
                pat = pat.masked_fill(~tril, 0.0)
                vpre = at.c_v(xin).float() - corr * (hc @ DELTA.T)
                v = vpre.view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v).float()
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            ao = at.c_proj(y.reshape(B, T, D).to(at.c_proj.weight.dtype))
        else:
            ao, v1n = block_attn(blk, xin, B, v1)
            if v1 is None:
                v1 = v1n
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if tgt and kind == 'mlp':
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / x.float().norm(dim=-1, keepdim=True)
            corr = lam0 * g
            hc = hsrc - STATE['mu'][Lsrc]
            pl = blk.mlp.Left(z).float() - corr * (hc @ DELTA['l'].T)
            pr = blk.mlp.Right(z).float() - corr * (hc @ DELTA['r'].T)
            mo = blk.mlp.Down((pl * pr).to(blk.mlp.Down.weight.dtype)) \
                + blk.mlp.Down_bias
            x = x + mo
        else:
            if DELTA is not None and L == Lsrc:
                hsrc = mlp_hidden(blk, z)
            x = x + blk.mlp(z)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()

    acc1 = {L: torch.zeros(HD, device=DEV) for L in LRANGE}
    acc2 = {L: torch.zeros(HD, device=DEV) for L in LRANGE}
    n0 = 0
    for i in range(0, 480, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for L, blk in enumerate(H):
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            ao, v1n = block_attn(blk, xin, idx.shape[0], v1)
            if v1 is None:
                v1 = v1n
            x = xm + ao
            z = F.rms_norm(x, (D,))
            if L in acc1:
                h = mlp_hidden(blk, z).reshape(-1, HD)
                acc1[L] += h.sum(0); acc2[L] += (h * h).sum(0)
            x = x + blk.mlp(z)
        n0 += idx.shape[0] * T
    for L in LRANGE:
        STATE['mu'][L] = acc1[L] / n0
        STATE['rms'][L] = (acc2[L] / n0).clamp_min(1e-12).sqrt()
    print("stats pass done", flush=True)

    def lowrank_delta(M, rmsv):
        Mw = M * rmsv.unsqueeze(0)
        U, S, V = torch.svd_lowrank(Mw, q=64, niter=4)
        Mr = (U[:, :32] * S[:32]) @ V[:, :32].T
        return M - Mr / rmsv.unsqueeze(0)

    def ce_run(Lsrc, kind, DELTA):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, Lsrc, kind, DELTA).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    clean = ce_run(0, 'val', None)
    res = {'clean': round(clean, 4)}
    print(f"clean {clean:.4f}", flush=True)
    SIG = {}; LOWR = {}
    for Lsrc in LRANGE:
        blk1 = H[Lsrc + 1]
        Wd = H[Lsrc].mlp.Down.weight.float().to(DEV)
        rmsv = STATE['rms'][Lsrc]
        EDGES = {
            'pat': {nm: mod.weight.float().to(DEV) @ Wd for nm, mod in
                    (('q', blk1.attn.c_q), ('k', blk1.attn.c_k),
                     ('q2', blk1.attn.c_q2), ('k2', blk1.attn.c_k2))},
            'val': blk1.attn.c_v.weight.float().to(DEV) @ Wd,
            'mlp': {'l': blk1.mlp.Left.weight.float().to(DEV) @ Wd,
                    'r': blk1.mlp.Right.weight.float().to(DEV) @ Wd},
        }
        for kind, C in EDGES.items():
            cut = ce_run(Lsrc, kind, C)
            if kind == 'val':
                beyond = lowrank_delta(C, rmsv)
            elif kind == 'pat':
                beyond = {nm: lowrank_delta(M, rmsv) for nm, M in C.items()}
            else:
                beyond = {nm: lowrank_delta(M, rmsv) for nm, M in C.items()}
            b32 = ce_run(Lsrc, kind, beyond)
            key = f'L{Lsrc}_{kind}'
            SIG[key] = round(cut - clean, 4)
            LOWR[key] = round(b32 - clean, 4)
            res[key] = {'signal': SIG[key], 'beyond_r32': LOWR[key]}
            print(f"{key}: signal {SIG[key]} beyond_r32 {LOWR[key]}", flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
        del EDGES
        torch.cuda.empty_cache()

    import statistics
    sigs = list(SIG.values())
    med = statistics.median(sigs)
    big = [k for k in SIG if SIG[k] >= 0.005]
    frac_lr = (sum(1 for k in big if LOWR[k] <= 0.30 * SIG[k]) / max(len(big), 1))
    mlp_top = sum(1 for L in LRANGE
                  if SIG[f'L{L}_mlp'] >= max(SIG[f'L{L}_pat'], SIG[f'L{L}_val']))
    pa = med >= 0.01
    pb = frac_lr >= 0.80
    pc = mlp_top / len(LRANGE) >= 0.60
    out = {'res': res, 'median_signal': round(med, 4),
           'frac_lowrank': round(frac_lr, 3), 'n_big': len(big),
           'mlp_largest_frac': round(mlp_top / len(LRANGE), 3),
           'pred_a_median_01': bool(pa), 'pred_b_lowrank_80': bool(pb),
           'pred_c_mlp_largest_60': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"median {med} lowrank {frac_lr:.2f} mlp_top {mlp_top}/{len(LRANGE)}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
