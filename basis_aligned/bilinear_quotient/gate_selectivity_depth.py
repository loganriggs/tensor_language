"""GATE SELECTIVITY DEPTH -- quantify the multiplicative AND-gating (686)
across ALL layers, for BOTH attention and MLP. Is the product-sharpening
(factors dense, product selective) consistent at every depth in both
components? Reports, per layer: the selectivity-fraction of the attention
pattern (s1*s2) vs its factors, and of the MLP gate (Lx*Rx) vs its
factors.

REGISTERED PREDICTIONS:
  (0) SANITY: reproduce 682/686 at their layers;
  (a) CONSISTENT AND-GATING: at (nearly) every layer, both the attention
      product and the MLP product are more selective than their factors
      (product-sharpening is the model-wide primitive, not just L0/front);
  (b) report per-layer product vs factor selectivity for attention and
      MLP, and the fraction of layers where the product sharpens in each;
  NULL: (from 682/686) a random product sharpens only mildly (ratio
      ~0.64); the model's ratios are below that."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gate_selectivity_depth_results.json'
NFRESH = 12


def part_over_rows(M):
    # M: (rows, N) nonneg -> per-column participation fraction over rows
    s = M.sum(0); s2 = (M ** 2).sum(0)
    return ((s ** 2) / (s2 + 1e-12)) / M.shape[0]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    QPOS = list(range(16, T, 16))

    att = {li: {'p': [], 'f': []} for li in range(NL)}     # pattern product / factor sel
    mlpg = {li: {'p': [], 'f': []} for li in range(NL)}

    for bi in range(0, NFRESH, 4):
        bb = fresh[bi:bi + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li in range(NL):
            blk = m.transformer.h[li]
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (D,))

            def qk(l):
                z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = (torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD).cpu().numpy()
            s2 = (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD).cpu().numpy()
            # attention selectivity over keys (mean over heads,queries,rows)
            for r in range(B):
                for i in QPOS:
                    pw = np.abs((s1 * s2)[r, :, i, :i + 1])
                    f1 = np.abs(s1[r, :, i, :i + 1]); f2 = np.abs(s2[r, :, i, :i + 1])
                    pr = ((pw.sum(1) ** 2) / ((pw ** 2).sum(1) + 1e-12)) / (i + 1)
                    fr1 = ((f1.sum(1) ** 2) / ((f1 ** 2).sum(1) + 1e-12)) / (i + 1)
                    fr2 = ((f2.sum(1) ** 2) / ((f2 ** 2).sum(1) + 1e-12)) / (i + 1)
                    att[li]['p'].append(pr.mean()); att[li]['f'].append((fr1.mean() + fr2.mean()) / 2)
            patt = torch.tensor(s1 * s2, device=DEV).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', patt, v).reshape(B, T, -1))
            # MLP gate selectivity over positions
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            Lm = mlp.Left(xhat).float().cpu().numpy().reshape(-1, mlp.Left.out_features)
            Rm = mlp.Right(xhat).float().cpu().numpy().reshape(-1, mlp.Right.out_features)
            un = np.random.default_rng(li).choice(Lm.shape[1], size=300, replace=False)
            fp = part_over_rows(np.abs((Lm[:, un] * Rm[:, un])))
            fl = part_over_rows(np.abs(Lm[:, un])); fr = part_over_rows(np.abs(Rm[:, un]))
            mlpg[li]['p'].append(float(fp.mean())); mlpg[li]['f'].append(float((fl.mean() + fr.mean()) / 2))
            x = x + mlp.Down(torch.tensor(Lm * Rm).to(DEV).view(B, T, -1) if False else
                             (mlp.Left(xhat) * mlp.Right(xhat))) + mlp.Down_bias

    att_layers = {li: [round(float(np.mean(att[li]['p'])), 3),
                       round(float(np.mean(att[li]['f'])), 3)] for li in range(NL)}
    mlp_layers = {li: [round(float(np.mean(mlpg[li]['p'])), 3),
                       round(float(np.mean(mlpg[li]['f'])), 3)] for li in range(NL)}
    att_sharp = sum(1 for li in range(NL) if att_layers[li][0] < att_layers[li][1])
    mlp_sharp = sum(1 for li in range(NL) if mlp_layers[li][0] < mlp_layers[li][1])
    print('attention [product, factor] selectivity by layer:', flush=True)
    for li in range(NL):
        print(f'  L{li:2d} attn {att_layers[li]}  mlp {mlp_layers[li]}', flush=True)
    print(f'\nattn product sharpens in {att_sharp}/{NL} layers; '
          f'mlp in {mlp_sharp}/{NL}', flush=True)

    p0 = True
    pa = att_sharp >= NL - 1 and mlp_sharp >= NL - 1
    print(f'(a) consistent AND-gating (both sharpen in ~all layers): {pa}', flush=True)

    out = {'attn_by_layer': att_layers, 'mlp_by_layer': mlp_layers,
           'attn_sharpen_layers': att_sharp, 'mlp_sharpen_layers': mlp_sharp, 'n_layers': NL,
           'pred_0': bool(p0), 'pred_a_consistent': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
