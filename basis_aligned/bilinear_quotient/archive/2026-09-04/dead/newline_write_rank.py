"""NEWLINE WRITE RANK -- confirm 612's low-rank-over-directions finding
is GENERAL by testing the newline decision, the symmetric closer to
611 (which confirmed unit-diffuseness general for newline).

612 found mlp0's article write is diffuse over units but LOW-RANK (~16
directions carry 84%) over its 1152-dim output space. Is the newline
write the same -- diffuse over units (611) but low-rank over
directions? If yes, "diffuse over units, compact in ~16 output
directions" is the general structure of this model's early decisions
(matching 569's "compact directions, distributed sources" and 589's
rank-16 attn0 write), confirmed on two independent decisions.

Method: keep only the top-r PCA directions of mlp0's output (mean-fill
the rest), measure the newline-margin shift. Whole-mlp0 newline shift
was +0.0505 (611); full-rank is identity.

Method: capture mlp0's output (the write into the residual) over real
data. SVD it to get its principal output directions. For each rank r,
replace mlp0's output with its rank-r projection (mean + top-r PCA
components, the rest collapsed to the mean), run the full model, and
measure the article-margin shift at article positions. The smallest r
that keeps the margin close to the true value is the article
decision's DIMENSIONALITY in mlp0's output. A small r => the decision
is direction-localized (diffuse over units but low-rank over
directions); a large r => genuinely high-rank/distributed.

Contrast baselines: mean-filling ALL of mlp0 (rank 0) is 608's +0.015
whole-layer shift; keeping full rank (1152) is 0 (identity). The
curve between them shows how many output directions the article
decision needs.

REGISTERED PREDICTIONS:
  (0) ENDPOINTS: rank-0 (mean-fill all) reproduces 608's whole-mlp0
      shift (~+0.015 within 0.01); full-rank reproduces the true
      margin (shift < 1e-3) -- VOIDS on failure;
  (a) LOW-RANK (the finding): report the smallest r whose article-
      margin shift is under 20% of the whole-layer (rank-0) shift.
      Prediction: r <= 64 -- the article decision is carried by a
      small number of output directions even though it is diffuse
      over units (parallel to attn0's rank-16, 589);
  (b) THE CURVE: report the shift at r = 1,2,4,8,16,32,64,128,256,512
      -- the shape is the result;
  (c) BEATS RANDOM DIRECTIONS: at the r from (a), the top-r PCA
      projection preserves the margin better than r RANDOM orthogonal
      directions (3 draws) -- the low rank is real structure, not any
      r-dim subspace;
  NULL: random-direction projections degrade monotonically worse than
      the PCA projection as r falls -- a random control that does not
      degrade is not a control."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_write_rank_results.json'
NFRESH = 64
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NL1, NL2, PER, QUO = 198, 628, 13, 1  # newline vs sentence-end


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    mlp = m.transformer.h[LJ].mlp

    # capture mlp0 output over all positions
    cap = []
    hk = mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0)                  # (N, D) mlp0 outputs
    mu = O.mean(0)
    Oc = O - mu
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    V = Vt.to(DEV)                          # (D, D) output PCA directions
    mu_d = mu.to(DEV)
    print(f'captured {O.shape[0]} mlp0 outputs; top singular values '
          f'{[round(float(s),1) for s in S[:5]]}', flush=True)

    nxt = fresh[:, 1:257].reshape(-1)
    art = ((nxt == NL1) | (nxt == NL2) |
           (nxt == PER) | (nxt == QUO)).numpy()
    amask = torch.tensor(art)

    def project_hook(r, basis=None):
        """Replace mlp0 output y with mu + P_r (y - mu), P_r = top-r
        projector in `basis` (default PCA V)."""
        Vr = (V[:r] if basis is None else basis[:r])       # (r, D)
        def fh(mo, i_, o_):
            y = o_.float()
            yc = y - mu_d
            proj = (yc @ Vr.T) @ Vr
            return (mu_d + proj).to(o_.dtype)
        return mlp.register_forward_hook(fh)

    def meanfill_hook():
        def fh(mo, i_, o_):
            return mu_d.expand_as(o_).to(o_.dtype)
        return mlp.register_forward_hook(fh)

    def margin(hook=None):
        hk = hook() if hook is not None else None
        out = []
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,))
            x0 = x
            v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            mg = p[..., NL1] + p[..., NL2] - p[..., PER] - p[..., QUO]
            out.append(mg.reshape(-1).cpu())
        if hk is not None:
            hk.remove()
        return torch.cat(out)

    base = float(margin(None)[amask].mean())
    rank0 = float(margin(meanfill_hook)[amask].mean()) - base   # = whole-layer
    full = float(margin(lambda: project_hook(D))[amask].mean()) - base
    p0 = abs(rank0 - 0.0505) < 0.015 and abs(full) < 1e-3
    print(f'baseline {base:.4f} | rank-0 (mean-fill) {rank0:+.4f} '
          f'(611 +0.0505) | full-rank {full:+.5f}: '
          f'{"HELD" if p0 else "FAILED -- VOID"}', flush=True)
    if not p0:
        json.dump({'void': 'endpoints failed', 'rank0': rank0, 'full': full},
                   open(OUT, 'w'), indent=1)
        return

    curve = {}
    for r in RANKS:
        curve[r] = round(float(margin(lambda r=r: project_hook(r))[amask].mean())
                         - base, 5)
        print(f'  rank {r:>4}: shift {curve[r]:+.5f} '
              f'({100*curve[r]/rank0:.0f}% of whole-layer)', flush=True)
        json.dump({'curve': curve, 'rank0': rank0}, open(OUT, 'w'), indent=1)

    thresh = 0.2 * abs(rank0)
    small_r = next((r for r in RANKS if abs(curve[r]) < thresh), None)
    pa = small_r is not None and small_r <= 64
    print(f'(a) smallest r with shift < 20% of whole-layer: {small_r} '
          f'(<=64?): {"HELD" if pa else "FAILED"}', flush=True)

    # random-direction control at small_r
    r_test = small_r if small_r else 64
    g = torch.Generator(device=DEV).manual_seed(0)
    rand_shifts = []
    for seed in range(3):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(D, r_test, generator=gg, device=DEV))
        basis = Q.T   # (r_test, D)
        rand_shifts.append(round(float(
            margin(lambda: project_hook(r_test, basis))[amask].mean()) - base, 5))
    rand_mean = float(np.mean([abs(x) for x in rand_shifts]))
    pca_abs = abs(curve[r_test]) if r_test in curve else abs(
        float(margin(lambda: project_hook(r_test))[amask].mean()) - base)
    pc = pca_abs < rand_mean
    print(f'(c) at r={r_test}: PCA |shift| {pca_abs:.5f} < random-dir '
          f'|shift| {rand_mean:.5f} (draws {rand_shifts}): '
          f'{"HELD" if pc else "FAILED"}', flush=True)

    out = {'baseline': base, 'rank0_wholelayer': rank0, 'full_rank': full,
           'pred_0': bool(p0), 'curve': curve, 'smallest_r_under_20pct': small_r,
           'pred_a': bool(pa), 'random_dir_shifts': rand_shifts,
           'pca_shift_at_rtest': pca_abs, 'pred_c': bool(pc),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
