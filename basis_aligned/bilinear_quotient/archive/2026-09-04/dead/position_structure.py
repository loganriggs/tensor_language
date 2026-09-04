"""POSITION STRUCTURE of MLP L1 (user: is the positional variable even/odd, or
early/late? how is it computed/separated?). mlp1's output has a causal position
subspace (776); position reaches mlp1 via attention's ROTARY embedding. Characterise
the position-conditional-mean table P (position -> mean mlp1 output over all tokens):
  (a) MONOTONIC / absolute (early vs late): variance explained by a smooth low-order
      polynomial in position;
  (b) PERIODIC (rotary sinusoids): FFT power spectrum along position -- dominant
      periods; fraction of variance at long periods (smooth) vs short (period<=4) vs
      period-2 (EVEN/ODD parity);
  (c) effective rank (how many position modes).
Answers: is the positional variable a smooth early/late signal + rotary long-period
sinusoids, or even/odd parity?

Runs on 384 rows (~98k tokens) for robust position means (default: more data).

REGISTERED PREDICTIONS:
  (0) SANITY: position means vary (nonzero centered variance);
  (a) SMOOTH / EARLY-LATE + LONG-PERIOD, NOT even/odd: >= 60% of the position-mean
      variance is in LONG periods (period > 8, i.e. smooth absolute-position +
      rotary low frequencies), and the period-2 (even/odd) share is small (< 10%);
      a low-order polynomial in position explains a large share (monotonic early/late
      component present);
  (b) report polynomial-fit R2, FFT band shares (period-2 / short / long), eff-rank;
  NULL: shuffling the position labels destroys the smooth/periodic structure (flat
      spectrum, low polynomial R2)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1; NPOS = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_structure_results.json'
NEVAL = 384; MINCOUNT = 20


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_posmeans(rows, n):
    ssum = torch.zeros(NPOS, D, device=DEV); scnt = torch.zeros(NPOS, device=DEV)
    cur = {'buf': None}
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cur.__setitem__('buf', o_.detach().float()))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        O = cur['buf']                                    # (b, T, D)
        T = O.shape[1]
        ssum[:T] += O.sum(0); scnt[:T] += O.shape[0]
    h.remove()
    keep = scnt >= MINCOUNT
    P = ssum[keep] / scnt[keep].clamp_min(1)[:, None]
    return P, torch.arange(NPOS, device=DEV)[keep]


def poly_r2(P, pos, deg):
    # variance of P explained by a degree-`deg` polynomial in position (per-dim, aggregate)
    x = pos.float()/pos.float().max()
    A = torch.stack([x**k for k in range(deg+1)], 1)      # (npos, deg+1)
    Pc = P - P.mean(0, keepdim=True)
    W = torch.linalg.lstsq(A, Pc).solution; Phat = A @ W
    return float((Phat**2).sum()/(Pc**2).sum().clamp_min(1e-9))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    P, pos = capture_posmeans(rows, NEVAL)
    Pc = P - P.mean(0, keepdim=True)
    npos = P.shape[0]
    print(f'{npos} positions with >= {MINCOUNT} samples', flush=True)

    # (a) polynomial (monotonic / smooth early-late)
    r2_lin = poly_r2(P, pos, 1); r2_cub = poly_r2(P, pos, 3)

    # (b) FFT along position axis (per dim), power spectrum
    Fp = torch.fft.rfft(Pc, dim=0)                        # (nfreq, D)
    power = (Fp.abs()**2).sum(1).cpu().numpy()            # power per frequency
    power[0] = 0                                          # drop DC (already centered)
    freqs = np.arange(len(power))                         # cycles over the npos window
    periods = np.where(freqs > 0, npos/np.maximum(freqs, 1), np.inf)
    tot = power.sum()
    long_share = float(power[(periods > 8)].sum()/max(tot, 1e-9))
    short_share = float(power[(periods <= 4) & (freqs > 0)].sum()/max(tot, 1e-9))
    # period-2 = the highest frequency (Nyquist, freq = npos/2)
    p2_share = float(power[-1]/max(tot, 1e-9))
    dom = int(freqs[1:][np.argmax(power[1:])]); dom_period = float(npos/max(dom, 1))
    print(f'(a) polynomial-in-position R2: linear {r2_lin:.3f}  cubic {r2_cub:.3f}', flush=True)
    print(f'(b) FFT: long-period(>8) share {long_share:.3f} | short(<=4) {short_share:.3f} | period-2/even-odd {p2_share:.4f} | dominant period {dom_period:.1f}', flush=True)

    er = float((torch.linalg.svdvals(Pc)**2).sum()**2/((torch.linalg.svdvals(Pc)**2)**2).sum())
    print(f'    effective rank of position-mean table {er:.1f} (of {npos} positions)', flush=True)

    # null: shuffle position order
    g = torch.Generator(device=DEV).manual_seed(0); perm = torch.randperm(npos, generator=g, device=DEV)
    r2_null = poly_r2(P[perm], pos, 3)
    print(f'    NULL shuffled-position cubic R2 {r2_null:.3f}', flush=True)

    p0 = float((Pc**2).sum()) > 0
    pa = long_share >= 0.6 and p2_share < 0.10 and r2_cub > 0.2
    null_ok = r2_null < 0.1
    out = {'n_positions': npos, 'poly_r2_linear': round(r2_lin, 4), 'poly_r2_cubic': round(r2_cub, 4),
           'fft_long_share': round(long_share, 4), 'fft_short_share': round(short_share, 4),
           'fft_period2_evenodd_share': round(p2_share, 5), 'dominant_period': round(dom_period, 1),
           'eff_rank': round(er, 2), 'null_cubic_r2': round(r2_null, 4),
           'pred_0': bool(p0), 'pred_a_smooth_earlylate_not_evenodd': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) positional variable is SMOOTH early/late + long-period (not even/odd): {pa}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
