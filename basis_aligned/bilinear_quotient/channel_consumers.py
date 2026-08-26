# channel_consumers: WHO CONSUMES THE mlp0 CHANNEL? (red-team arm 2 of the circuit
# work: does the discovered structure PREDICT the consumers?) Remove the weight-
# channel top-8 subspace at mlp0's output (S1484 grain fix), run clean vs removed on
# the same rows, and rank ALL 162 heads by relative output change ||dy_h|| / ||y_h||
# (y_h = the head's pre-c_proj output slice). Structural predictions under test:
# the ATLAS says mlp0's edges are front-loaded (blocks 1-4), so propagation should
# hit early heads hardest; the ROSTER says named specialists are the functionally
# special heads — enrichment in the affected set tests whether that list predicts
# consumers of an mlp0 perturbation.
#
# Registered predictions:
#   pred_a >= 6 of the top-10 affected heads sit in layers 1-4 (front-loaded wiring
#          predicts propagation).
#   pred_b named-roster heads are >= 2x base-rate enriched in the top-30 (base rate
#          30/162 = 18.5% -> bar: >= 12 of 30).
#   pred_c split-half Spearman rank correlation of the 162 head deltas >= .7.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'channel_consumers_results.json'
NR = 240
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ROSTER = {(5, 7), (8, 1), (8, 2), (8, 3), (8, 7), (10, 2), (10, 3), (10, 4),
          (10, 5), (10, 6), (13, 0), (13, 5), (13, 8), (14, 4), (14, 6), (14, 7),
          (16, 0), (16, 3), (16, 4), (16, 5), (17, 0), (17, 1), (17, 2)}


@torch.no_grad()
def block_attn_y(blk, xin, B, v1):
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
    return y, at.c_proj(y.reshape(B, T, D)), v1


@torch.no_grad()
def fwd_heads(idx, remove, MU, RMS, W8, MEAN_OUT=None):
    """Return {(L,h): y_h [B,T,128]} for all heads; remove: channel-8 cut at mlp0."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; ys = {}
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    for L, blk in enumerate(H):
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        y, ao, v1n = block_attn_y(blk, xin, B, v1)
        if v1 is None:
            v1 = v1n
        for hh in range(9):
            ys[(L, hh)] = y[:, :, hh, :].float()
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0 and remove:
            h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
            hw = (h0 - MU) / RMS
            comp_h = ((hw @ W8.T) @ W8) * RMS
            x = x + (blk.mlp(z).float() - comp_h @ Wd0.T).to(x.dtype)
        else:
            x = x + blk.mlp(z)
    return ys


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    # h0 stats + channel basis (as in channel_circuit)
    FR = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 480, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        _, ao, _ = block_attn_y(blk, xin, idx.shape[0], None)
        z = F.rms_norm(xm + ao, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    MU = a1 / n0
    RMS = (a2 / n0).clamp_min(1e-12).sqrt()
    at1 = H[1].attn
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    STACK = torch.cat([at1.c_q.weight.float().to(DEV) @ Wd0,
                       at1.c_k.weight.float().to(DEV) @ Wd0,
                       at1.c_q2.weight.float().to(DEV) @ Wd0,
                       at1.c_k2.weight.float().to(DEV) @ Wd0,
                       at1.c_v.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Left.weight.float().to(DEV) @ Wd0,
                       H[1].mlp.Right.weight.float().to(DEV) @ Wd0], 0)
    _, _, Vt = torch.linalg.svd(STACK * RMS.unsqueeze(0), full_matrices=False)
    W8 = Vt[:8]
    print("basis built", flush=True)

    dsum = {}; bsum = {}
    dsum2 = {}; halves = [({}, {}), ({}, {})]
    for i in range(0, NR, 8):
        idx = EVR[i:i + 8, :-1].to(DEV).contiguous()
        yc = fwd_heads(idx, False, MU, RMS, W8)
        yr = fwd_heads(idx, True, MU, RMS, W8)
        half = 0 if i < NR // 2 else 1
        for key in yc:
            d = float((yr[key] - yc[key]).norm())
            b = float(yc[key].norm())
            dsum[key] = dsum.get(key, 0.0) + d
            bsum[key] = bsum.get(key, 0.0) + b
            hd, hb = halves[half]
            hd[key] = hd.get(key, 0.0) + d
            hb[key] = hb.get(key, 0.0) + b
        del yc, yr
    rel = {key: dsum[key] / max(bsum[key], 1e-9) for key in dsum}
    order = sorted(rel, key=lambda k2: -rel[k2])
    top10 = order[:10]; top30 = order[:30]
    print("top10:", [(f"{L}.{h}", round(rel[(L, h)], 4)) for L, h in top10],
          flush=True)

    n_early = sum(1 for (L, h) in top10 if 1 <= L <= 4)
    n_roster = sum(1 for k2 in top30 if k2 in ROSTER)
    r0 = {k2: halves[0][0][k2] / max(halves[0][1][k2], 1e-9) for k2 in rel}
    r1 = {k2: halves[1][0][k2] / max(halves[1][1][k2], 1e-9) for k2 in rel}
    import statistics
    keys = list(rel)
    rank0 = {k2: i for i, k2 in enumerate(sorted(keys, key=lambda x: -r0[x]))}
    rank1 = {k2: i for i, k2 in enumerate(sorted(keys, key=lambda x: -r1[x]))}
    n = len(keys)
    dsq = sum((rank0[k2] - rank1[k2]) ** 2 for k2 in keys)
    rho = 1 - 6 * dsq / (n * (n * n - 1))

    pa = n_early >= 6
    pb = n_roster >= 12
    pc = rho >= 0.7
    out = {'top10': [[f'{L}.{h}', round(rel[(L, h)], 4)] for L, h in top10],
           'top30_roster_count': n_roster, 'top10_early_count': n_early,
           'spearman_split_half': round(rho, 3),
           'per_layer_mean_rel': {L: round(sum(rel[(L, h)] for h in range(9)) / 9, 4)
                                  for L in range(18)},
           'pred_a_early_6_of_10': bool(pa), 'pred_b_roster_2x': bool(pb),
           'pred_c_stable_rho_7': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"early {n_early}/10 roster {n_roster}/30 rho {rho:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
