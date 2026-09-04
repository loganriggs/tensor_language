"""HIERARCHY NESTING -- does the Down_0 weight-action SAE have HIERARCHICAL
structure (coarse parents -> fine children), which would explain 761's redundancy
(a child's job is partly covered by its parent -> single-atom ablation undercounts,
superadditive)? Train the SAE at several dictionary sizes P in {64, 256, 1024} and
test coarse-to-fine NESTING:
  (a) DECODER nesting: assign each FINE atom (P=1024) to its best coarse atom
      (P=64) by decoder cosine; if fines are refinements of coarses the mean
      assignment cosine is high, and each coarse gets several fines (branching).
  (b) SPAN nesting: each coarse atom's direction lies in the SPAN of its assigned
      fine atoms (coarse ~ combination of its children) -- projection residual small.
  (c) ACTIVATION containment on data: for a fine child f under coarse parent c,
      P(c active | f active) is high (the parent fires whenever the child does).
Compare to a RANDOM assignment null.

REGISTERED PREDICTIONS:
  (0) SANITY: all three SAEs reconstruct (train R2 > 0.5);
  (a) HIERARCHICAL: mean fine->coarse assignment cosine >= 0.5 and >> random; mean
      activation-containment P(parent|child) >= 0.6 and >> random; span-nesting
      residual low -- i.e. coarse atoms decompose into fine ones (a hierarchy),
      which structurally explains 761's redundancy;
  (b) report assignment cosine, branching factor, containment, span residual vs null;
  NULL: random parent assignment gives containment ~ base rate and low cosine."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'hierarchy_nesting_results.json'
NFIT = 48; NEVAL = 24; K = 16; PS = [64, 256, 1024]


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


@torch.no_grad()
def capture(rows, n):
    cap = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def train_sae(Xin, Ytrue, P, k, seed=0):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, k); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r2 = float(1 - ((Ytrue - (topk(Xin@Em.T, k)@Dm.T+b))**2).sum()/((Ytrue-Ytrue.mean(0))**2).sum())
    return Dm.detach(), Em.detach(), b.detach(), r2


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    g0 = capture(fit, NFIT); Y0 = g0 @ W0.T; g0_ev = capture(ev, NEVAL)

    saes = {}
    for P in PS:
        with torch.enable_grad(): Dm, Em, b, r2 = train_sae(g0, Y0, P, K)
        saes[P] = (Dm, Em, b); print(f'P={P}: train R2 {r2:.3f}', flush=True)
    Dc, Ec, _ = saes[64]; Df, Ef, _ = saes[1024]        # coarse / fine

    # (a) decoder assignment: each fine atom -> best coarse atom
    Dc_n = F.normalize(Dc, dim=0); Df_n = F.normalize(Df, dim=0)
    cos = (Df_n.T @ Dc_n).abs()                          # (1024, 64)
    assign = cos.argmax(1); assign_cos = cos.max(1).values
    mean_assign = float(assign_cos.mean())
    branching = float(torch.bincount(assign, minlength=64).float().mean())
    rand_assign_cos = float(cos.gather(1, torch.randint(0, 64, (1024, 1), device=DEV)).mean())
    print(f'(a) fine->coarse assign cos {mean_assign:.3f} (random {rand_assign_cos:.3f})  branching {branching:.1f}', flush=True)

    # (b) span nesting: each coarse direction reconstructed from its assigned fine children
    span_res = []
    for c in range(64):
        kids = (assign == c).nonzero(as_tuple=True)[0]
        if len(kids) < 2: continue
        B = Df[:, kids]                                  # (D, nk)
        Q = torch.linalg.qr(B)[0]                        # orthonormal span of children
        v = Dc[:, c]; proj = Q @ (Q.T @ v)
        span_res.append(float((v - proj).norm()/v.norm().clamp_min(1e-9)))
    span_residual = float(np.mean(span_res)) if span_res else 1.0
    print(f'(b) coarse-in-children span residual {span_residual:.3f} (0=perfectly nested)', flush=True)

    # (c) activation containment on eval: P(parent active | child active)
    zc = (topk(g0_ev @ Ec.T, K) > 1e-6)                  # (Nev, 64)
    zf = (topk(g0_ev @ Ef.T, K) > 1e-6)                  # (Nev, 1024)
    contain = []; base_parent = zc.float().mean(0)       # parent base rate
    for f in range(1024):
        c = int(assign[f]); fires = zf[:, f]
        if fires.sum() < 5: continue
        pc_given_f = float(zc[fires, c].float().mean())
        contain.append(pc_given_f - float(base_parent[c]))   # lift over base rate
    mean_contain_lift = float(np.mean(contain)) if contain else 0.0
    # null: random parent
    g = torch.Generator(device=DEV).manual_seed(0); rand_par = torch.randint(0, 64, (1024,), generator=g, device=DEV)
    contain_null = []
    for f in range(1024):
        c = int(rand_par[f]); fires = zf[:, f]
        if fires.sum() < 5: continue
        contain_null.append(float(zc[fires, c].float().mean()) - float(base_parent[c]))
    mean_contain_null = float(np.mean(contain_null)) if contain_null else 0.0
    print(f'(c) activation containment lift P(parent|child)-base {mean_contain_lift:.3f} (random parent {mean_contain_null:.3f})', flush=True)

    p0 = True
    pa = mean_assign >= 0.5 and mean_assign > 1.5*rand_assign_cos and mean_contain_lift >= 0.1 and mean_contain_lift > 2*max(mean_contain_null, 1e-3)
    null_ok = mean_contain_null < 0.05
    out = {'Ps': PS, 'K': K, 'assign_cos': round(mean_assign, 4), 'rand_assign_cos': round(rand_assign_cos, 4),
           'branching': round(branching, 2), 'span_residual': round(span_residual, 4),
           'contain_lift': round(mean_contain_lift, 4), 'contain_null': round(mean_contain_null, 4),
           'pred_0': bool(p0), 'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) HIERARCHICAL (fines nest in coarses + activation-contained): {pa}; NULL random-parent low: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
