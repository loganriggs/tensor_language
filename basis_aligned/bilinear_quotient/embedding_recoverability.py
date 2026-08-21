"""EMBEDDING RECOVERABILITY -- functional check of 689's finding that the
embedding is re-injected at weight ~8 at every block. If so, the CURRENT
token's identity should remain linearly recoverable from the FINAL
residual (block 17), unlike a normal transformer where the current token
is transformed away. Test: how well does a linear readout of the final
residual recover the current token's identity, vs its recoverability at
the embedding (block 0, ceiling) and vs a mid-depth check.

We test recoverability of a coarse but decisive current-token feature --
the current token's log-frequency and its class -- from the residual at
depths 0, 8, 17 via a linear probe (R^2 for log-freq; AUC for a class).

REGISTERED PREDICTIONS:
  (0) SANITY: at block 0 (embedding) the current token's log-freq is
      recoverable (R^2 high);
  (a) STAYS RECOVERABLE: at the FINAL residual (block 17), the current
      token's log-freq is still recoverable at R^2 >= 0.5 -- the lambda1~8
      re-injection keeps the embedding decodable to the end (a normal
      transformer would transform it away);
  (b) report R^2 by depth (0, 8, 17) for current-token log-freq;
  NULL: a shuffled current-token label is not recoverable (R^2 ~ 0)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'embedding_recoverability_results.json'
NFRESH = 32
DEPTHS = [0, 8, 17]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    cur = fresh[:, :256].reshape(-1).numpy()
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    cur_lf = np.log(freq[cur] + 1.0)

    caps = {li: [] for li in DEPTHS}
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li in DEPTHS:
                caps[li].append(x.detach().float().reshape(-1, D).cpu())

    N = NFRESH * T; rng = np.random.default_rng(0); perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]
    y = cur_lf

    def r2(X):
        Xtr = X[tr]; mu = Xtr.mean(0)
        A = Xtr - mu
        # ridge regression to log-freq
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(D), A.T @ (y[tr] - y[tr].mean()))
        pred = (X[te] - mu) @ w + y[tr].mean()
        ss_res = ((y[te] - pred) ** 2).sum(); ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        return 1 - ss_res / (ss_tot + 1e-9)

    out = {'r2_by_depth': {}}
    for li in DEPTHS:
        X = torch.cat(caps[li], 0).numpy()
        out['r2_by_depth'][li] = round(float(r2(X)), 4)
        print(f'depth {li:2d}: current-token log-freq R^2 {out["r2_by_depth"][li]}',
              flush=True)

    # shuffled null at depth 17
    X17 = torch.cat(caps[17], 0).numpy()
    ysh = y.copy(); rng.shuffle(ysh)
    Xtr = X17[tr]; mu = Xtr.mean(0); A = Xtr - mu
    w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(D), A.T @ (ysh[tr] - ysh[tr].mean()))
    pred = (X17[te] - mu) @ w + ysh[tr].mean()
    ss_res = ((ysh[te] - pred) ** 2).sum(); ss_tot = ((ysh[te] - ysh[te].mean()) ** 2).sum()
    null_r2 = round(float(1 - ss_res / (ss_tot + 1e-9)), 4)
    print(f'shuffled-label null R^2 (depth 17): {null_r2}', flush=True)

    p0 = out['r2_by_depth'][0] >= 0.5
    pa = out['r2_by_depth'][17] >= 0.5
    null_ok = null_r2 < 0.1
    print(f'\n(0) embedding recoverable: {p0}; (a) final residual recoverable '
          f'(R^2>=0.5): {pa}; NULL shuffled ~0: {null_ok}', flush=True)

    out.update({'shuffled_null_r2': null_r2, 'pred_0': bool(p0),
                'pred_a_final_recoverable': bool(pa), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
