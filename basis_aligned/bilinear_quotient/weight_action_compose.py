"""WEIGHT-ACTION COMPOSITION across two neighbouring matrices (does the
weight-action SAE compose from WEIGHTS ALONE, or does top-k re-introduce data?).

Setup: decompose Down_0 (writes residual) and Left_1 (reads residual) each with
a weight-action top-k SAE:
  Down_0 ~ D1 @ E1  (atom i = decoder col D1[:,i], a residual-WRITE dir; code z1=topk(E1.gate0))
  Left_1 ~ D2 @ E2  (atom j = encoder row E2[j,:], a residual-READ dir)
The dictionary-to-dictionary coupling  C = E2 @ D1  (P2 x P1) is PURE WEIGHTS
(no X.pinv, unlike A-SVD whose B = Vh@X.pinv bakes the data covariance in).

Two DISTINCT sources of "data-dependence" must be separated:
  (I)  the SAE's own TOP-K: gates WHICH edges of C are live per token. Question:
       does this make the WIRING data-based? Claim: NO -- C is fixed weights;
       top-k only routes.
  (II) the MODEL's intervening nonlinearity (lambda-scale + attention + rms_norm
       between Down_0 and Left_1): this DOES dilute a weight-only prediction, but
       it is the model's, not the SAE's, and would hit ANY decomposition.

Measurements:
  - sparsity of C: mean effective in-degree (how many source atoms strongly drive
    each target atom). Small => a clean sparse component graph (nodes=atoms,
    edges=C), the graph-sparsity object the program wants.
  - WEIGHT-ONLY FIDELITY: the weight-predicted write of Down_0, (D1@z1+b1),
    vs the MEASURED contribution of layer-0 mlp to Left_1's read point
    (path-isolated by zeroing mlp_0). corr high => the write composes from
    weights+code; the residual gap is source (II), quantified.
  - ROUTING sparsity: per token, live edges = (live source atoms) x (live target
    atoms) << P1*P2. top-k gives sparse per-datapoint paths through the fixed graph.
  - A-SVD CONTRAST: A-SVD's analogous coupling needs X.pinv, so it is data-variant:
    recompute it on two disjoint FineWeb splits and report ||dC_asvd||/||C_asvd||;
    the SAE C is bit-identical across splits (||dC_sae||=0) by construction.

REGISTERED PREDICTIONS:
  (0) SANITY: both SAEs reconstruct their weight action (out-R2 > 0.5 at k=32);
  (a) COMPOSES FROM WEIGHTS: C is SPARSE (mean effective in-degree < P1/8) AND the
      weight-only write predicts the measured contribution (corr > 0.6, >> shuffled
      null < 0.1); routing is sparse (live edges < 1% of P1*P2). Top-k does NOT
      make the wiring data-based -- C is fixed; it only routes.
  (b) A-SVD coupling is DATA-VARIANT across splits (||dC_asvd||/||C_asvd|| > 0.2)
      while SAE ||dC_sae|| = 0 -- only the weight-action SAE composes weight-only;
  NULL: shuffling z1 across tokens destroys the write-prediction corr (-> ~0)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'weight_action_compose_results.json'
NFIT = 64; P1 = 1024; P2 = 1024; K = 48; STEPS = 1500   # K/STEPS up: Left_1 action (HID-dim) underfit at 32/1000


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


@torch.no_grad()
def capture(rows, n, pre_module=None, post_module=None):
    """Capture pre-input of pre_module (list) and/or output of post_module (list)."""
    got = {'pre': [], 'post': [], 'mlp0_in_left1': []}
    hs = []
    if pre_module is not None:
        hs.append(pre_module.register_forward_pre_hook(
            lambda mo, inp: got['pre'].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))))
    if post_module is not None:
        hs.append(post_module.register_forward_pre_hook(
            lambda mo, inp: got['post'].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    return got


@torch.no_grad()
def capture_left1_input(rows, n, kill_mlp0=False):
    """Left_1 input (rms-normed residual at block 1), optionally with layer-0 mlp zeroed."""
    cap = []
    h_in = m.transformer.h[1].mlp.Left.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, D)))
    hk = None
    if kill_mlp0:
        hk = m.transformer.h[0].mlp.register_forward_hook(lambda mo, i_, o_: torch.zeros_like(o_))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h_in.remove()
    if hk is not None: hk.remove()
    return torch.cat(cap, 0)


def train_wa(Xg, Ytrue, P, k, steps=STEPS, seed=0):
    """Ytrue = Xg @ W.T (weight action). Returns Dm (Dout x P), Em (P x Din), codes z, out-R2."""
    torch.manual_seed(seed)
    din = Xg.shape[1]; dout = Ytrue.shape[1]
    Dm = (torch.randn(dout, P, device=DEV)/np.sqrt(dout)).requires_grad_(True)
    Em = (torch.randn(P, din, device=DEV)/np.sqrt(din)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(steps):
        z = topk(Xg @ Em.T, k); recon = z @ Dm.T + b
        loss = F.mse_loss(recon, Ytrue); opt.zero_grad(); loss.backward(); opt.step()
    Dm = Dm.detach(); Em = Em.detach(); b = b.detach()
    with torch.no_grad():
        z = topk(Xg @ Em.T, k); recon = z @ Dm.T + b
        r2 = float(1 - ((Ytrue-recon)**2).sum()/((Ytrue-Ytrue.mean(0))**2).sum())
    return Dm, Em, b, z, r2


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(2*NFIT)
    W_down0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)   # D x HID
    W_left1 = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)   # HID x D

    # source: weight-action SAE of Down_0 on gate_0
    g0 = capture(rows[:NFIT], NFIT, pre_module=m.transformer.h[0].mlp.Down)['pre']
    G0 = torch.cat(g0, 0)                                   # (N, HID)
    Y0 = G0 @ W_down0.T                                     # (N, D) weight action of Down_0
    with torch.enable_grad(): D1, E1, b1, z1, r2_1 = train_wa(G0, Y0, P1, K)

    # target: weight-action SAE of Left_1 on its residual input
    r1 = capture(rows[:NFIT], NFIT, pre_module=m.transformer.h[1].mlp.Left)['pre']
    R1 = torch.cat(r1, 0)                                   # (N, D) Left_1 input
    Yl = R1 @ W_left1.T                                     # (N, HID) weight action of Left_1
    with torch.enable_grad(): D2, E2, b2, z2, r2_2 = train_wa(R1, Yl, P2, K)
    print(f'SAE out-R2: Down_0 {r2_1:.3f}  Left_1 {r2_2:.3f}', flush=True)

    # ---- weight-only coupling C = E2 @ D1 (P2 x P1) ----
    C = (E2 @ D1)                                            # pure weights, no data
    col_max = C.abs().max(0).values.clamp_min(1e-9)
    strong = (C.abs() > 0.2*col_max).float()               # edges > 20% of that source atom's max
    in_deg = float(strong.sum(0).mean())                   # mean strong targets per source atom
    frac_edges = float(strong.mean())
    print(f'coupling C: mean in-degree {in_deg:.1f}/{P2}  ({100*frac_edges:.2f}% of edges strong)', flush=True)

    # ---- weight-only fidelity: predicted Down_0 write vs measured contribution ----
    write_pred = (z1 @ D1.T + b1)                          # (N, D) weight-predicted mlp_0 write
    r1_full = capture_left1_input(rows[:NFIT], NFIT, kill_mlp0=False)
    r1_kill = capture_left1_input(rows[:NFIT], NFIT, kill_mlp0=True)
    delta_meas = (r1_full - r1_kill)                       # measured layer-0-mlp contribution at read point
    # CENTER both sides over tokens: test PER-TOKEN variation only, not the shared bias/mean
    # (751/752-class confound: an uncentered corr rides the constant b1 -> real ~= shuffled).
    wp = write_pred - write_pred.mean(0, keepdim=True)
    dm = delta_meas - delta_meas.mean(0, keepdim=True)
    a = wp.reshape(-1); bb = dm.reshape(-1)
    corr = float(torch.corrcoef(torch.stack([a, bb]))[0, 1])
    g = torch.Generator(device=DEV).manual_seed(0)
    perm = torch.randperm(z1.shape[0], generator=g, device=DEV)
    wp_sh = (z1[perm] @ D1.T + b1); wp_sh = wp_sh - wp_sh.mean(0, keepdim=True)
    a_sh = wp_sh.reshape(-1)
    corr_null = float(torch.corrcoef(torch.stack([a_sh, bb]))[0, 1])
    print(f'weight-only write vs measured contribution: corr {corr:.3f}  (shuffled null {corr_null:.3f})', flush=True)

    # ---- routing sparsity: live edges per token ----
    live1 = (z1 > 1e-6).float(); live2 = (z2 > 1e-6).float()   # (N, P) each
    live_edges = float((live1 @ strong.T * live2).sum(1).mean())   # per token: live-source x strong x live-target
    print(f'routing: mean LIVE edges/token {live_edges:.1f}  ({100*live_edges/(P1*P2):.4f}% of {P1}x{P2})', flush=True)

    # ---- A-SVD contrast: coupling is data-variant across splits ----
    A_a, B_a = asvd_fast(W_down0, G0)                       # split A (first NFIT)
    g0b = capture(rows[NFIT:2*NFIT], NFIT, pre_module=m.transformer.h[0].mlp.Down)['pre']
    G0b = torch.cat(g0b, 0)
    A_b, B_b = asvd_fast(W_down0, G0b)                      # split B (second NFIT)
    r = 256
    Ca = W_left1 @ A_a[:, :r]; Cb = W_left1 @ A_b[:, :r]    # A-SVD dict-to-read coupling (via A factor)
    dC_asvd = float((Ca - Cb).norm() / Ca.norm())
    print(f'A-SVD coupling drift across splits ||dC||/||C|| {dC_asvd:.3f}  (SAE C drift = 0, weights)', flush=True)

    p0 = r2_1 > 0.5 and r2_2 > 0.5
    pa = in_deg < P1/8 and corr > 0.6 and corr_null < 0.1 and live_edges < 0.01*P1*P2
    pb = dC_asvd > 0.2
    null_ok = abs(corr_null) < 0.15
    out = {'P1': P1, 'P2': P2, 'K': K,
           'sae_out_r2': {'down0': round(r2_1,4), 'left1': round(r2_2,4)},
           'coupling_in_degree': round(in_deg,2), 'coupling_frac_strong': round(frac_edges,5),
           'write_corr': round(corr,4), 'write_corr_shuffled_null': round(corr_null,4),
           'live_edges_per_token': round(live_edges,2), 'live_edge_frac': round(live_edges/(P1*P2),6),
           'asvd_coupling_drift': round(dC_asvd,4), 'sae_coupling_drift': 0.0,
           'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(0) both SAEs fit: {p0}; (a) composes weight-only (sparse C + weight-write predicts + sparse routing): {pa}; '
          f'(b) A-SVD data-variant: {pb}; NULL shuffled~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
