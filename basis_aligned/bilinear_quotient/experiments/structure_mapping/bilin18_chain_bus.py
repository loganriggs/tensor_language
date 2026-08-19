"""The 3->4 edge, given variables: how many channels does the chain's bus have?

§28 oriented the edge (layer 4 reads layers 2-3; layer 3 is the stronger supplier) but
an edge without variables is not an abstraction. Weights-first, the natural candidate
variables are the top eigenvectors of the COUPLING OPERATOR

    K(3->4) = C3^{1/2} G2(4) C3^{1/2}

where C3 is layer 3's output covariance (what 3 actually writes, with realistic
magnitudes) and G2(4) is layer 4's input-mode Λ-Gram (what 4's quadratic is sensitive
to, from weights + input second moment). K's top eigenvectors are "the directions layer
3 writes that layer 4 reads", computable in seconds before any intervention.

REGISTERED PREDICTIONS, before running:
  P1: patching layer 3's write along the top-k coupling directions (value transplanted
      from a different document) transfers MORE of the full-edge effect than patching
      k of layer 3's own top output-PCA directions -- i.e. the weights know what 4
      READS, which is different from what 3 writes loudest. Bar: coupling beats PCA at
      every k <= 8.
  P2: the edge is few-channel: 8 coupling directions (of 1152) carry over half of the
      full-L3-patch effect on layer 4's output. Bar: T(8) > 0.5.

MEASUREMENT. Base and source sequences from different documents. At layer 3's MLP
output, replace the write's component along a k-dim span with the source's values;
measure the change this induces in LAYER 4's OUTPUT, normalised by the change induced
by transplanting layer 3's entire write:

    T(k) = E||mo4(span-k patch) - mo4(base)||^2 / E||mo4(full patch) - mo4(base)||^2

Controls: k random directions; k of layer 3's top output PCs. The full patch is the
upper bound (T = 1 at k = 1152 by construction).
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot

NH, HD, D = 9, 128, 1152
KS = (1, 2, 4, 8, 16)
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_chain_bus_results.json')

PATCH3 = None      # (Q, source mo3 tensor) -> replace component along Q


@torch.no_grad()
def run_to4(idx):
    """Forward through layer 4; returns (mo3, mo4). PATCH3 applied to mo3."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    mo3 = mo4 = None
    for li in range(5):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
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
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if li == 3:
            if PATCH3 is not None:
                Q, src = PATCH3
                c_b = mo.float() @ Q
                c_s = src.float() @ Q
                mo = mo + ((c_s - c_b) @ Q.T).to(mo.dtype)
            mo3 = mo.detach()
        if li == 4:
            mo4 = mo.detach().float()
        x = x + mo
    return mo3, mo4


def main():
    global PATCH3
    t0 = time.time()
    base_rows = FW[300:324, :257].to(DEV)
    src_rows = FW[400:424, :257].to(DEV)

    # ---- weights-first: the coupling operator ----
    # layer-3 output covariance from the corpus
    accs = []
    for i in range(0, 96, 6):
        acc = []
        fwd(FW[i:i + 6, :513].to(DEV), collect=3, acc=acc)
        accs.append(acc[0])
    Y3 = torch.cat(accs)
    Y3c = Y3 - Y3.mean(0)
    C3 = Y3c.T @ Y3c / Y3c.shape[0]
    # layer-4 input second moment + Λ input-mode Gram (pure weights + S)
    ins = []

    def hook(mod, inp, o):
        ins.append(inp[0].detach().reshape(-1, D).float())

    h = m.transformer.h[4].mlp.register_forward_hook(hook)
    for i in range(0, 60, 6):
        b = FW[i:i + 6, :513].to(DEV)
        m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    h.remove()
    X4 = torch.cat(ins)
    S4 = X4.T @ X4 / X4.shape[0]
    mlp4 = m.transformer.h[4].mlp
    L = mlp4.Left.weight.detach().float()
    R = mlp4.Right.weight.detach().float()
    Dw = mlp4.Down.weight.detach().float()
    DD = Dw.T @ Dw
    G24 = L.T @ (DD * (R @ S4 @ R.T)) @ L + R.T @ (DD * (L @ S4 @ L.T)) @ R
    ev3, U3 = torch.linalg.eigh(C3.double())
    ev3 = ev3.clamp_min(0)
    C3h = ((U3 * ev3.sqrt()) @ U3.T).float()
    K = C3h @ G24 @ C3h
    evK, UK = torch.linalg.eigh(K.double())
    idx = evK.argsort(descending=True)
    coup = torch.linalg.qr(C3h.double() @ UK[:, idx[:16]])[0].float()  # back to output space, orthonormal
    print('coupling operator built (weights + S + C3); predictions frozen\n')

    # PCA and random control spans
    _, _, Vh3 = torch.linalg.svd(Y3c.float(), full_matrices=False)
    pca = orth(Vh3[:16].T)
    g = torch.Generator(device=DEV).manual_seed(0)
    rnd = orth(torch.randn(D, 16, device=DEV, generator=g))

    # ---- interventions ----
    PATCH3 = None
    mo3_b, mo4_b = run_to4(base_rows)
    _, mo4_bsrc = run_to4(src_rows)
    mo3_s, _ = run_to4(src_rows)
    # full-edge reference: transplant layer 3's entire write
    PATCH3 = (torch.eye(D, device=DEV), mo3_s)
    _, mo4_full = run_to4(base_rows)
    PATCH3 = None
    denom = float((mo4_full - mo4_b).pow(2).mean())
    print(f'full-L3-write transplant moves layer-4 output by {denom:.4f} '
          f'(the T=1 reference)\n')
    out = {'denom': denom, 'curves': {}}
    print(f"  {'k':>4} {'coupling T(k)':>14} {'L3-PCA T(k)':>12} {'random T(k)':>12}")
    for k in KS:
        row = {}
        for tag, span in (('coupling', coup[:, :k]), ('pca', pca[:, :k]),
                          ('random', rnd[:, :k])):
            PATCH3 = (span, mo3_s)
            _, mo4_p = run_to4(base_rows)
            PATCH3 = None
            row[tag] = float((mo4_p - mo4_b).pow(2).mean()) / denom
        out['curves'][k] = row
        print(f"  {k:>4} {row['coupling']:>14.3f} {row['pca']:>12.3f} "
              f"{row['random']:>12.3f}", flush=True)

    p1 = all(out['curves'][k]['coupling'] > out['curves'][k]['pca']
             for k in KS if k <= 8)
    p2 = out['curves'][8]['coupling'] > 0.5
    out['P1_coupling_beats_pca'] = bool(p1)
    out['P2_T8_over_half'] = bool(p2)
    print(f"\nP1 (coupling beats L3's own PCA at k<=8): "
          f"{'HELD' if p1 else 'FAILED'}")
    print(f"P2 (8 of 1152 channels carry half the edge): "
          f"{'HELD' if p2 else 'FAILED'}  (T(8) = {out['curves'][8]['coupling']:.3f})")

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
