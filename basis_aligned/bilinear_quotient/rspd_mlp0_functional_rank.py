"""RSPD MLP0 FUNCTIONAL RANK -- run the same CE-priced A-SVD (694) + core
readout (696) on mlp0, the FRONT class-decider (634), for a front-vs-back
functional-rank comparison. mlp0's Down output affects ALL downstream
blocks (not just the readout), so CE is measured by substituting the rank-r
surrogate into the live model and running the full forward -- the pricing
already handles this.

Same method: A-SVD on Down (1152x4608) with X = real bilinear-gate input;
rank-r surrogate priced by real CE; smallest r for 80% of mlp0's benefit;
random-projection null; then name the top-4 core directions by unembedding.
NOTE: mlp0's core directions written into the residual are read by 17 more
blocks before the unembedding, so the direct-unembedding readout is a
rougher proxy than for mlp17 (flagged, not overclaimed).

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank A-SVD reproduces baseline CE (|d|<0.01);
  (a) LOW-RANK CORE EXISTS: mlp0's 20/80 rank r80 <= 128 (a data-
      conditioned low-rank core exists in the front too);
  (b) FRONT-VS-BACK: report mlp0's r80 next to mlp17's r80=4. Register the
      expectation that the FRONT is HIGHER-rank than the back (r80_mlp0 >
      4): the class decision spans many token-classes, vs mlp17's
      calibration+few-writers;
  NULL: random rank-r projection recovers ~0/negative (r80 = never)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
HID = 4608
LAYER = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp0_functional_rank_results.json'
NFRESH = 24
NCAP = 12
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@torch.no_grad()
def ce_with_down(fresh, Wsub):
    mlp = m.transformer.h[LAYER].mlp
    orig = mlp.Down.weight.data
    if Wsub == 'ablate':
        mlp.Down.weight.data = torch.zeros_like(orig)
    elif Wsub is not None:
        mlp.Down.weight.data = Wsub.to(orig.dtype).to(orig.device)
    ce_s = 0.0; n = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean')
        ce_s += float(ce) * idx.shape[0]; n += idx.shape[0]
    mlp.Down.weight.data = orig
    return ce_s / n


@torch.no_grad()
def capture_gate(fresh):
    cap = []
    mlp = m.transformer.h[LAYER].mlp
    h = mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, NCAP, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().cpu()
    X = capture_gate(fresh)
    print(f'gate X {tuple(X.shape)}, Down W {tuple(W.shape)}', flush=True)

    A_fac, B_fac = generate_lowrank_approximation(W, X, target=X @ W.T)
    maxr = A_fac.shape[1]

    ce_full = ce_with_down(fresh, None)
    ce_ablate = ce_with_down(fresh, 'ablate')
    benefit = ce_ablate - ce_full
    print(f'CE_full {ce_full:.4f}  CE_ablate {ce_ablate:.4f}  benefit {benefit:.4f}',
          flush=True)

    W_fullrank = (A_fac @ B_fac).float()
    ce_fullrank = ce_with_down(fresh, W_fullrank)
    p0 = abs(ce_fullrank - ce_full) < 0.01
    print(f'(0) full-rank A-SVD CE {ce_fullrank:.4f}: {"HELD" if p0 else "FAILED"}',
          flush=True)

    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    rows = {'asvd': {}, 'random': {}}
    for r in RANKS:
        if r > maxr:
            break
        W_r = (A_fac[:, :r] @ B_fac[:r, :]).float()
        rec = (ce_ablate - ce_with_down(fresh, W_r)) / benefit
        Qr = Q[:, :r]
        rec_rand = (ce_ablate - ce_with_down(fresh, (Qr @ (Qr.T @ W)).float())) / benefit
        rows['asvd'][r] = round(float(rec), 4); rows['random'][r] = round(float(rec_rand), 4)
        print(f'r={r:4d}: A-SVD recovered {rec:.3f} | random {rec_rand:.3f}', flush=True)

    def r80(tbl):
        for r in RANKS:
            if r in tbl and tbl[r] >= 0.80:
                return r
        return None
    r80_a = r80(rows['asvd']); r80_r = r80(rows['random'])

    # name the top-4 core directions (rough proxy: 17 blocks downstream)
    W_U = m.lm_head.weight.data.float().cpu()
    V = W_U.shape[0]
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    log_freq = np.log(freq + 1.0); valid = freq > 0
    core = A_fac[:, :4]; core = core / core.norm(dim=0, keepdim=True)
    def d1(t):
        try: return cl.d1(int(t))
        except Exception: return f'<{t}>'
    dirs = []
    for j in range(4):
        shift = (W_U @ core[:, j]).numpy(); order = np.argsort(-shift)
        corr = float(np.corrcoef(shift[valid], log_freq[valid])[0, 1])
        dd = {'j': j, 'freq_corr': round(corr, 3),
              'boost': [d1(t) for t in order[:8]],
              'suppress': [d1(t) for t in order[::-1][:8]]}
        dirs.append(dd)
        print(f'dir {j}: freq_corr {corr:+.3f} boost {dd["boost"][:5]}', flush=True)

    pa = r80_a is not None and r80_a <= 128
    null_ok = (r80_r is None) or (r80_a is not None and r80_r >= 2 * r80_a)
    front_higher = r80_a is not None and r80_a > 4
    print(f'\nmlp0 r80={r80_a} (mlp17 r80=4); random r80={r80_r}', flush=True)
    print(f'(a) low-rank core (r80<=128): {pa}; front>back (r80>4): {front_higher}; '
          f'NULL: {null_ok}', flush=True)

    out = {'ce_full': round(ce_full, 4), 'ce_ablate': round(ce_ablate, 4),
           'benefit': round(benefit, 4), 'recovered_by_rank': rows,
           'r80_asvd': r80_a, 'r80_random': r80_r, 'r80_mlp17': 4,
           'core_directions': dirs, 'pred_0': bool(p0), 'pred_a_low_rank': bool(pa),
           'front_higher_rank': bool(front_higher), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
