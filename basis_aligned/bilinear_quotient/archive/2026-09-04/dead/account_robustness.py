"""ACCOUNT ROBUSTNESS -- consolidation check for a wrap-up. Re-measure
the account's HEADLINE quantitative claims on a FRESH, LARGER held-out
corpus (64 rows, ~4x the usual probe size) to confirm they are not
artifacts of the small samples used to establish them. This is due
diligence for publishing, not a new thread. Three headline numbers:

  (A) massive-activation peak: the final residual is sharply peaked (a few
      dims >> median). Prior: ~58x (676/691 on 24 rows).
  (B) AND-gating product-sharpening in attention: the product s1*s2 is
      more focal (participation ~0.23) than either factor (~0.54). Prior:
      682/684 -- product sharpens in ~all layers.
  (C) embedding recoverability at the final residual: current-token log-
      freq linearly recoverable R^2 ~0.73 (690 on 32 rows).

REGISTERED PREDICTIONS (robustness = prior numbers reproduce within band):
  (A) final-residual peak > 20x (massive dims robust);
  (B) attention product participation < factor participation in >= 16/18
      layers (AND-gating robust);
  (C) embedding recoverability R^2 >= 0.5 at final residual (robust);
  NULL: shuffled-label recoverability ~0 (< 0.1)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'account_robustness_results.json'
NFRESH = 64


def part_over_rows(M):
    s = M.sum(0); s2 = (M ** 2).sum(0)
    return ((s ** 2) / (s2 + 1e-12)) / M.shape[0]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)
    V = m.lm_head.weight.shape[0]
    cur = fresh[:, :256].reshape(-1).numpy()
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    y = np.log(freq[cur] + 1.0)

    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    QPOS = list(range(16, T, 16))
    att = {li: {'p': [], 'f': []} for li in range(NL)}
    ss_fin = torch.zeros(D, dtype=torch.float64); n = 0
    fin_cap = []

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
            for r in range(B):
                for i in QPOS:
                    pw = np.abs((s1 * s2)[r, :, i, :i + 1])
                    f1 = np.abs(s1[r, :, i, :i + 1]); f2 = np.abs(s2[r, :, i, :i + 1])
                    pr = ((pw.sum(1) ** 2) / ((pw ** 2).sum(1) + 1e-12)) / (i + 1)
                    fr1 = ((f1.sum(1) ** 2) / ((f1 ** 2).sum(1) + 1e-12)) / (i + 1)
                    fr2 = ((f2.sum(1) ** 2) / ((f2 ** 2).sum(1) + 1e-12)) / (i + 1)
                    att[li]['p'].append(pr.mean()); att[li]['f'].append((fr1.mean() + fr2.mean()) / 2)
            mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            patt = torch.tensor(s1 * s2, device=DEV).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', patt, v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        ss_fin += (x.float() ** 2).reshape(-1, D).sum(0).double().cpu()
        fin_cap.append(x.detach().float().reshape(-1, D).cpu())
        n += idx.numel()

    # (A) peak
    rfin = np.sqrt((ss_fin / n).numpy())
    peak = float(rfin.max() / (np.median(rfin) + 1e-9))
    # (B) AND-gating
    sharp = sum(1 for li in range(NL)
                if np.mean(att[li]['p']) < np.mean(att[li]['f']))
    prod_mean = float(np.mean([np.mean(att[li]['p']) for li in range(NL)]))
    fac_mean = float(np.mean([np.mean(att[li]['f']) for li in range(NL)]))
    # (C) recoverability
    X = torch.cat(fin_cap, 0).numpy()
    N = X.shape[0]; rng = np.random.default_rng(0); perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]

    def r2(yy):
        Xtr = X[tr]; mu = Xtr.mean(0); A = Xtr - mu
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(D), A.T @ (yy[tr] - yy[tr].mean()))
        pred = (X[te] - mu) @ w + yy[tr].mean()
        return float(1 - ((yy[te] - pred) ** 2).sum() / (((yy[te] - yy[te].mean()) ** 2).sum() + 1e-9))
    r2_fin = r2(y)
    ysh = y.copy(); rng.shuffle(ysh); r2_null = r2(ysh)

    print(f'(A) final-residual peak (max/median):     {peak:.1f}x', flush=True)
    print(f'(B) AND-gating: product<factor in {sharp}/{NL} layers '
          f'(prod {prod_mean:.3f} vs factor {fac_mean:.3f})', flush=True)
    print(f'(C) embedding recoverability R^2 (final): {r2_fin:.3f}  '
          f'(shuffled null {r2_null:.3f})', flush=True)
    pA = peak > 20; pB = sharp >= 16; pC = r2_fin >= 0.5; nullok = r2_null < 0.1
    print(f'\n(A) peak>20x: {pA}; (B) sharpen>=16/18: {pB}; '
          f'(C) R^2>=0.5: {pC}; NULL: {nullok}', flush=True)
    print(f'ALL HEADLINE NUMBERS ROBUST: {pA and pB and pC and nullok}', flush=True)

    out = {'peak': round(peak, 2), 'and_gating_sharpen_layers': sharp, 'n_layers': NL,
           'attn_product_part': round(prod_mean, 4), 'attn_factor_part': round(fac_mean, 4),
           'recover_r2_final': round(r2_fin, 4), 'recover_null_r2': round(r2_null, 4),
           'pred_A_peak': bool(pA), 'pred_B_andgating': bool(pB),
           'pred_C_recover': bool(pC), 'null_ok': bool(nullok),
           'all_robust': bool(pA and pB and pC and nullok), 'n_rows': NFRESH,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
