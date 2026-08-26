# edge_axis_causal: CAUSAL TEST OF THE NAMED AXES (S1470 named D1 'determiner/NP-
# start' and D2 'capitalized name-fragment' from the composed-edge SVDs; naming is
# only real if cutting the axis hurts the PREDICTIONS the name implies). Cuts are
# rank-1 mean-preserving: remove the whitened component of (h0 - mu0) along each
# edge's own dir (pattern dir0+values dir0 = D1; pattern dir1+values dir1 = D2;
# random whitened unit vector = control), applied at block-1's pattern AND values
# reads. Scored globally AND class-conditionally: CE at positions whose PREVIOUS
# token is in the axis's top-token set (determiners for D1, name-fragments for D2).
# Original header: MEAN-PRESERVING EDGE CUTS (S1466: the values edge cut costs
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
#   pred_a D1 cut: determiner-following CE rise >= 3x its global CE rise (the name
#          predicts WHERE the damage lands).
#   pred_b control cut: global rise <= .003 (specificity).
#   pred_c D2 cut: name-fragment-following CE rise >= 2x its global rise.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'edge_axis_causal_results.json'
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
    MU0['mu'] = a1 / n0
    rms = (a2 / n0).clamp_min(1e-12).sqrt()
    print("mu, rms measured", flush=True)

    PATC = torch.cat([CP[n] for n in ('q', 'k', 'q2', 'k2')], 0)
    DIRS = {}
    for nm, M in (('pat', PATC), ('val', CV)):
        _, _, Vt = torch.linalg.svd(M * rms.unsqueeze(0), full_matrices=False)
        DIRS[nm] = Vt[:2]
    gr = torch.Generator().manual_seed(99)
    rnd = torch.randn(HD, generator=gr).to(DEV)
    rnd = rnd / rnd.norm()
    AXES = {'D1': {'pat': DIRS['pat'][0], 'val': DIRS['val'][0]},
            'D2': {'pat': DIRS['pat'][1], 'val': DIRS['val'][1]},
            'ctrl': {'pat': rnd, 'val': rnd}}
    AX = {'ax': None}

    import tiktoken
    ENC = tiktoken.get_encoding('gpt2')
    DET = [' the', ' our', ' his', ' a', ' your', ' their', ' an', ' my', ' this',
           'The', ' The', ' its', ' her']
    FRAG = [' Ch', ' Pl', ' Sh', ' H', ' G', ' Br', ' Th', ' B', ' T', ' W', ' M',
            ' D', ' L', ' R', ' Fl', ' Bl', ' Sp', ' Z', ' K', ' F', ' S', ' Howard']
    DET_IDS = torch.tensor([ENC.encode(t)[0] for t in DET])
    FRAG_IDS = torch.tensor([ENC.encode(t)[0] for t in FRAG])

    def axis_cut(hc, edge):
        d = AXES[AX['ax']][edge]
        s = (hc / rms) @ d
        return s.unsqueeze(-1) * (d * rms)

    @torch.no_grad()
    def fwd_axis(idx, on):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]; h0 = None
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            if L == 1 and on:
                lam0 = float(blk.lambdas[0])
                g = (D ** 0.5) / xm.float().norm(dim=-1, keepdim=True)
                corr = lam0 * g
                hc = h0 - MU0['mu']
                cut_p = axis_cut(hc, 'pat')
                cut_v = axis_cut(hc, 'val')
                pre = {}
                for nm2, mod_ in (('q', at.c_q), ('k', at.c_k),
                                  ('q2', at.c_q2), ('k2', at.c_k2)):
                    p = mod_(xin).float() - corr * (cut_p @ CP[nm2].T)
                    pre[nm2] = p.view(B, T, 9, 128)
                cos, sin = at.rotary(pre['q'])
                q = are(F.rms_norm(pre['q'], (128,)), cos, sin)
                k = are(F.rms_norm(pre['k'], (128,)), cos, sin)
                q2 = are(F.rms_norm(pre['q2'], (128,)), cos, sin)
                k2 = are(F.rms_norm(pre['k2'], (128,)), cos, sin)
                pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
                    * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
                tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
                pat = pat.masked_fill(~tril, 0.0)
                vpre = at.c_v(xin).float() - corr * (cut_v @ CV.T)
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

    def ce_run(on):
        s_ = 0.0; n_ = 0
        sd = 0.0; nd = 0; sf = 0.0; nf = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_axis(idx, on).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
            isdet = torch.isin(idx.cpu(), DET_IDS).to(DEV) & mk
            isfrag = torch.isin(idx.cpu(), FRAG_IDS).to(DEV) & mk
            sd += float(ce[isdet].sum()); nd += int(isdet.sum())
            sf += float(ce[isfrag].sum()); nf += int(isfrag.sum())
        return s_ / max(n_, 1), sd / max(nd, 1), sf / max(nf, 1), nd, nf

    res = {}
    g0, d0, f0, nd, nf = ce_run(False)
    res['clean'] = {'global': round(g0, 4), 'det': round(d0, 4),
                    'frag': round(f0, 4), 'n_det': nd, 'n_frag': nf}
    print(f"clean {res['clean']}", flush=True)
    for ax in ('D1', 'D2', 'ctrl'):
        AX['ax'] = ax
        g1, d1, f1, _, _ = ce_run(True)
        res[ax] = {'global': round(g1, 4), 'det': round(d1, 4), 'frag': round(f1, 4),
                   'rise_global': round(g1 - g0, 4), 'rise_det': round(d1 - d0, 4),
                   'rise_frag': round(f1 - f0, 4)}
        print(f"{ax}: {res[ax]}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    pa = res['D1']['rise_det'] >= 3 * max(res['D1']['rise_global'], 1e-6)
    pb = res['ctrl']['rise_global'] <= 0.003
    pc = res['D2']['rise_frag'] >= 2 * max(res['D2']['rise_global'], 1e-6)
    out = {'res': res, 'pred_a_D1_det_3x': bool(pa),
           'pred_b_ctrl_le_003': bool(pb), 'pred_c_D2_frag_2x': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
