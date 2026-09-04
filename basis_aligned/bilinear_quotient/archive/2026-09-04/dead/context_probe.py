"""CONTEXT PROBE of MLP L1 residual (properly-powered follow-up to 774, which was
inconclusive because the unsupervised mean-subspace method is swamped by finite-
sample noise for the weaker previous-token signal). Use a SUPERVISED held-out ridge
probe: regress the previous token's EMBEDDING from mlp1's context residual (current-
token subspace removed), held-out R^2 vs a shuffled-previous-token null. This
cleanly answers whether previous-token information is DECODABLE from the residual --
and, combined with 774's finding that removing the prev-token subspace costs only
0.015 nats, whether it is a DECODABLE-BUT-CAUSALLY-INERT signal (the read!=write
pattern, FINDINGS 2) or genuinely absent.

For scale, also probe CURRENT token (should be high) and POSITION (sanity).

REGISTERED PREDICTIONS:
  (0) SANITY: current-token embedding is strongly decodable from the FULL output
      (held-out R^2 > 0.5); position weakly or not;
  (a) PREV-TOKEN DECODABLE BUT WEAK: prev-token embedding is decodable from the
      context residual above the shuffled null (held-out R^2 >= 0.1 and >> null),
      resolving 774 -- the info is PRESENT (so 774's null=signal was underpower,
      not absence), but per 774 it is causally near-inert (read!=write);
  (b) report held-out R^2 for prev-token (residual) vs current-token (full) vs
      position, each with shuffled null;
  NULL: shuffling the target labels drops held-out R^2 to ~0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'context_probe_results.json'
NEVAL = 64; MINCOUNT = 5; RSEM = 64; RIDGE = 1e-1


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, n):
    cap = []; cur = []; prev = []; pos = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        forward_logits(idx)
        c = idx.cpu().numpy(); p = np.full_like(c, -1); p[:, 1:] = c[:, :-1]
        pp = np.broadcast_to(np.arange(c.shape[1]), c.shape)
        cur.append(c.reshape(-1)); prev.append(p.reshape(-1)); pos.append(pp.reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(cur), np.concatenate(prev), np.concatenate(pos)


def mean_subspace(O, labels, r=RSEM):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def ridge_r2(X, Y, ridge=RIDGE, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g, device=X.device)
    ntr = int(X.shape[0]*0.7); tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    mx = Xtr.mean(0, keepdim=True); my = Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; A = Xc.T @ Xc; A.diagonal().add_(ridge*float(A.diagonal().mean()))
    W = torch.linalg.solve(A, Xc.T @ (Ytr - my))
    Yhat = (Xte - mx) @ W + my
    ss_res = ((Yte - Yhat)**2).sum(); ss_tot = ((Yte - Yte.mean(0, keepdim=True))**2).sum()
    return float(1 - ss_res/ss_tot.clamp_min(1e-9))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, cur, prev, pos = capture(rows, NEVAL)
    wte = m.transformer.wte.weight.data.float().to(DEV)                # (V, D)

    # context residual: remove current-token subspace
    Ucur = mean_subspace(O, cur); Rc = O - O.mean(0, keepdim=True)
    resid = Rc - (Rc @ Ucur) @ Ucur.T
    valid = prev >= 0
    Xr = resid[torch.from_numpy(valid).to(DEV)]
    Ofull = O[torch.from_numpy(valid).to(DEV)]
    prev_emb = wte[torch.from_numpy(prev[valid]).to(DEV)]              # (Nv, D)
    cur_emb = wte[torch.from_numpy(cur[valid]).to(DEV)]
    pos_t = torch.from_numpy(pos[valid]).float().to(DEV)[:, None]

    g = np.random.RandomState(0)
    def with_null(X, Y):
        r = ridge_r2(X, Y); Ysh = Y[torch.from_numpy(g.permutation(Y.shape[0])).to(DEV)]
        return round(r, 4), round(ridge_r2(X, Ysh), 4)

    r_prev, n_prev = with_null(Xr, prev_emb)            # prev-token from RESIDUAL
    r_cur, n_cur = with_null(Ofull, cur_emb)            # current-token from FULL output
    r_pos, n_pos = with_null(Xr, pos_t)                 # position from residual
    r_prev_full, _ = with_null(Ofull, prev_emb)         # prev-token from FULL output (for scale)
    print(f'prev-token from residual: R2 {r_prev} (null {n_prev}) | from full {r_prev_full}', flush=True)
    print(f'current-token from full : R2 {r_cur} (null {n_cur})', flush=True)
    print(f'position from residual  : R2 {r_pos} (null {n_pos})', flush=True)

    p0 = r_cur > 0.5
    pa = r_prev >= 0.1 and r_prev - n_prev >= 0.05
    null_ok = abs(n_prev) < 0.05
    out = {'prev_from_residual_r2': r_prev, 'prev_null': n_prev, 'prev_from_full_r2': r_prev_full,
           'cur_from_full_r2': r_cur, 'cur_null': n_cur, 'pos_from_residual_r2': r_pos, 'pos_null': n_pos,
           'pred_0': bool(p0), 'pred_a_prev_decodable': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) prev-token DECODABLE-but-weak from residual (>=0.1 & >>null): {pa}; NULL clean: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
