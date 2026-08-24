# reencode_subspace: WHAT does mlp3's re-encoding of the match evidence look like?
# §1243: mlp3 carries the main re-encoding (blind it and 1.29 of the write's 2.23-nat value
# dies). Characterize the re-encoding DELTA: for each position, delta = mlp3(rms(x)) -
# mlp3(rms(x - matcher_write)) under the real forward on repeat rows — the part of mlp3's
# output that exists because the match evidence does.
#
# Registered predictions:
#   pred_a COMPACT CODE: top-16 PCs of the per-position deltas carry >= 70% of their
#          variance (the re-encoding is low-rank, unlike the §1127 content construction).
#   pred_b GENUINE RE-ENCODING: mean |cos(delta, raw write)| <= 0.3 (a different direction,
#          not amplification of the same vector).
#   pred_c STABLE: split-half top-PC alignment |cos| >= 0.7 (a code, not noise); and the
#          delta is MATCH-GATED — mean ||delta|| on repeat rows >= 3x mean ||delta|| on
#          natural prose rows at the same positions (the code fires when matches exist).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reencode_subspace_results.json'
NR = 24; QSTART = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}


@torch.no_grad()
def deltas(idx):
    """Returns (P, Dw): per-position mlp3 re-encoding deltas and raw writes at scored pos."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    wsum = torch.zeros(B, T, D, device=DEV)
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
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
        if L in MATCHERS:
            for h in MATCHERS[L]:
                yh = torch.zeros_like(y)
                yh[:, :, h, :] = y[:, :, h, :]
                wsum = wsum + at.c_proj(yh.reshape(B, T, D)).float()
        x = xm + at.c_proj(y.reshape(B, T, D))
        if L == 3:
            full = blk.mlp(F.rms_norm(x, (D,)))
            blind = blk.mlp(F.rms_norm(x - wsum.to(x.dtype), (D,)))
            dl = (full - blind).float()[:, QSTART:]
            return dl.reshape(-1, D), wsum.float()[:, QSTART:].reshape(-1, D)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    PROSE = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    DL, W = [], []
    DLp = []
    for i in range(0, NR, 4):
        dl, w = deltas(REP[i:i + 4, :-1].to(DEV).contiguous())
        DL.append(dl.cpu()); W.append(w.cpu())
        dlp, _ = deltas(PROSE[i:i + 4, :-1].to(DEV).contiguous())
        DLp.append(dlp.cpu())
    DL = torch.cat(DL); W = torch.cat(W); DLp = torch.cat(DLp)

    Dc = DL - DL.mean(0)
    U, S, V = torch.pca_lowrank(Dc, q=64)
    var = (S ** 2); share16 = float(var[:16].sum() / ((Dc ** 2).sum()))
    cosw = F.cosine_similarity(DL, W, dim=-1)
    half = DL.shape[0] // 2
    _, _, Va = torch.pca_lowrank(Dc[:half], q=4)
    _, _, Vb = torch.pca_lowrank(Dc[half:], q=4)
    stab = float(F.cosine_similarity(Va[:, 0], Vb[:, 0], dim=0).abs())
    gate = float(DL.norm(dim=-1).mean() / DLp.norm(dim=-1).mean().clamp_min(1e-9))

    out = {'n_rows': NR, 'n_pos': int(DL.shape[0]),
           'top16_var_share': round(share16, 4),
           'mean_abs_cos_delta_write': round(float(cosw.abs().mean()), 4),
           'splithalf_pc1_cos': round(stab, 4),
           'repeat_vs_prose_norm_ratio': round(gate, 3),
           'pred_a_compact': bool(share16 >= 0.70),
           'pred_b_reencoded': bool(float(cosw.abs().mean()) <= 0.3),
           'pred_c_stable_gated': bool(stab >= 0.7 and gate >= 3.0),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top16 var {share16:.3f} | |cos(delta,write)| {float(cosw.abs().mean()):.3f} | split-half PC1 {stab:.3f} | gate {gate:.2f}x")
    print(f"pred_a compact {out['pred_a_compact']} | pred_b reencoded {out['pred_b_reencoded']} | pred_c stable+gated {out['pred_c_stable_gated']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
