"""FAITHFUL ROTATION (user: how to do MDL/SAE-style sparsity WITHOUT the
faithfulness cost of a lossy SAE). Idea: rotating the EXACT rank-K SVD basis
by an orthogonal R preserves the rank-K reconstruction EXACTLY (U_K and U_K@R
span the same subspace), but changes the basis. Choose R (varimax) to make
the per-datapoint codes SPARSER. This gets SAE-style sparse codes with EXACT
faithfulness (no separate lossy model).

Compare per-datapoint CE-based code length (rank components by CE-relevance
|coeff x CE-grad|, count to 90%) for mlp1's output basis:
  raw weight-SVD (top K) vs varimax-rotated weight-SVD (top K) vs A-SVD.
All three reconstruct the SAME rank-K subspace exactly (faithful); only the
basis (and thus per-datapoint code sparsity) differs.

REGISTERED PREDICTIONS:
  (0) SANITY: rotation preserves the subspace (reconstruction identical to
      raw SVD at rank K -- verify projector ||U U^T - U' U'^T|| ~ 0);
  (a) SPARSER CODES: the varimax-rotated basis gives SHORTER mean per-
      datapoint CE-code length than raw weight-SVD (a faithful sparsity win)
      -- report the mean code lengths;
  (b) report mean/median code length + usage Gini for raw / rotated / A-SVD;
  NULL: a RANDOM orthogonal rotation does NOT shorten codes vs raw (varimax's
      gain is from the sparsifying objective, not just any rotation)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'faithful_rotation_results.json'
NFIT = 48; NDATA = 48; K = 128; FRAC = 0.90


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


def varimax(Phi, q=80, tol=1e-6):
    # Phi: (p, k) loadings; returns orthogonal R (k,k) maximizing squared-loading variance
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        L = Phi @ R
        M = Phi.T @ (L**3 - (1.0/p) * L @ np.diag(np.diag(L.T @ L)))
        u, s, vh = np.linalg.svd(M)
        R = u @ vh; d_old = d; d = s.sum()
        if d_old != 0 and d/d_old < 1 + tol: break
    return R


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def capture_out_grad(rows, n):
    Os = []; Gs = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        store = {}
        h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo,i_,o_: store.__setitem__('o', o_.detach().requires_grad_(True)) or store['o'])
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        h.remove()
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').backward()
        o = store['o']; Os.append(o.detach().float().reshape(-1, D).cpu()); Gs.append(o.grad.float().reshape(-1, D).cpu())
        m.zero_grad(set_to_none=True)
    return torch.cat(Os,0), torch.cat(Gs,0)


def ce_code_lengths(O, G, U, frac=FRAC):
    C = (O @ U).numpy(); GU = (G @ U).numpy(); R = np.abs(C*GU)
    Rs = -np.sort(-R, axis=1); cum = np.cumsum(Rs, axis=1); tot = cum[:, -1:]+1e-12
    lengths = ((cum/tot) < frac).sum(1) + 1
    order = np.argsort(-R, axis=1); usage = np.zeros(U.shape[1])
    for i in range(R.shape[0]): usage[order[i, :lengths[i]]] += 1
    pp = np.sort(usage[usage>0]/(usage.sum()+1e-12))
    gini = float(1 - 2*np.sum(pp.cumsum())/(pp.sum()*len(pp)) + 1/len(pp)) if len(pp) else 0.0
    return lengths, gini


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, dat = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    Xg = capture_gate(fit, NFIT).to(DEV)
    Uw = torch.linalg.svd(W)[0][:, :K].cpu()                 # (D,K) exact rank-K basis
    A, _ = asvd_fast(W, Xg); Ua = (A[:, :K]/A[:, :K].norm(dim=0, keepdim=True)).cpu()

    Uw_np = Uw.numpy()
    Rvar = varimax(Uw_np)                                     # sparsifying rotation
    Uw_rot = torch.tensor(Uw_np @ Rvar).float()
    g = torch.Generator().manual_seed(1); Rr = torch.linalg.qr(torch.randn(K, K, generator=g))[0].numpy()
    Uw_rand = torch.tensor(Uw_np @ Rr).float()

    # faithfulness: same projector
    proj_raw = Uw_np @ Uw_np.T; proj_rot = (Uw_np@Rvar) @ (Uw_np@Rvar).T
    proj_diff = float(np.linalg.norm(proj_raw - proj_rot))

    O, Grad = capture_out_grad(dat, NDATA)
    res = {}
    for name, U in [('weight_svd_raw', Uw), ('weight_svd_varimax', Uw_rot),
                    ('weight_svd_randrot', Uw_rand), ('asvd', Ua)]:
        L, gini = ce_code_lengths(O, Grad, U)
        res[name] = {'mean_code_len': round(float(L.mean()),2), 'median': int(np.median(L)),
                     'usage_gini': round(gini,3)}
        print(f'{name:22s}: mean-len {res[name]["mean_code_len"]:.1f}  median {res[name]["median"]}  gini {gini:.2f}', flush=True)

    print(f'\nfaithfulness (||proj_raw - proj_varimax||): {proj_diff:.2e} (0 = exact same subspace)', flush=True)
    sparser = res['weight_svd_varimax']['mean_code_len'] < res['weight_svd_raw']['mean_code_len']
    null_ok = res['weight_svd_varimax']['mean_code_len'] <= res['weight_svd_randrot']['mean_code_len']
    print(f'(a) varimax sparser than raw: {sparser}; NULL varimax<=randrot: {null_ok}; faithful: {proj_diff<1e-3}', flush=True)
    out = {'layer': LAYER, 'K': K, 'proj_diff': proj_diff, 'bases': res,
           'pred_a_sparser': bool(sparser), 'null_ok': bool(null_ok), 'faithful': bool(proj_diff<1e-3),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
