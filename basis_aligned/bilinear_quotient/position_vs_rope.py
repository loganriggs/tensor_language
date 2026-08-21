"""POSITION vs ROPE (user: are the position-means MORE SIMILAR / lower-rank than
RoPE itself? does the MLP collapse RoPE's multi-frequency encoding into a coarse
position summary -- the position analog of the token->class collapse 780?). Compare
mlp1's position-conditional-mean table P (position -> mean output) against the RAW
RoPE encoding R (position -> [cos(p*f_i), sin(p*f_i)] over all rotary frequencies):
  (a) effective rank: P (786: ~2) vs R (high, ~ 2*num-frequencies) -- does the MLP
      output carry a MUCH lower-rank position representation than RoPE?
  (b) position-position similarity: are NEARBY positions more similar (smoother) in P
      than in R? (RoPE's high frequencies make nearby positions differ; a coarse
      early/late summary makes them nearly identical.)
  (c) RSA: correlate the position x position dissimilarity matrices of P and R -- how
      much does the MLP position representation reorganise RoPE's geometry?

Runs on 384 rows (more data).

REGISTERED PREDICTIONS:
  (0) SANITY: R has more than a few effective dims;
  (a) MLP COLLAPSES ROPE: eff-rank(P) << eff-rank(R) (P at least 5x lower), so the MLP
      output's position representation is far lower-rank than RoPE -- a coarse
      readout, not RoPE's full multi-frequency code;
  (b) SMOOTHER: adjacent-position cosine similarity is HIGHER in P than in R (the MLP
      position summary is smoother than RoPE);
  (c) report the P-vs-R RSA (relative-geometry correlation);
  NULL: n/a (descriptive comparison)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1; NPOS = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_vs_rope_results.json'
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
        O = cur['buf']; T = O.shape[1]; ssum[:T] += O.sum(0); scnt[:T] += O.shape[0]
    h.remove()
    keep = scnt >= MINCOUNT
    return (ssum[keep] / scnt[keep].clamp_min(1)[:, None]), torch.arange(NPOS, device=DEV)[keep]


def rope_table(positions):
    inv_freq = m.transformer.h[0].attn.rotary.inv_freq.float().to(DEV)   # (F,)
    p = positions.float()[:, None]                                       # (npos, 1)
    ang = p * inv_freq[None, :]                                          # (npos, F)
    return torch.cat([ang.cos(), ang.sin()], 1)                         # (npos, 2F)


def eff_rank(X):
    s2 = torch.linalg.svdvals(X - X.mean(0, keepdim=True))**2
    return float((s2.sum()**2)/(s2**2).sum())


def adjacent_sim(V):
    Vn = F.normalize(V - V.mean(0, keepdim=True), dim=1)
    return float((Vn[:-1] * Vn[1:]).sum(1).mean())


def rsa(A, B):
    An = F.normalize(A - A.mean(0, keepdim=True), dim=1); Bn = F.normalize(B - B.mean(0, keepdim=True), dim=1)
    RA = 1 - An @ An.T; RB = 1 - Bn @ Bn.T
    iu = torch.triu_indices(A.shape[0], A.shape[0], 1, device=A.device)
    a = RA[iu[0], iu[1]]; b = RB[iu[0], iu[1]]
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    P, pos = capture_posmeans(rows, NEVAL)
    R = rope_table(pos)
    erP = eff_rank(P); erR = eff_rank(R)
    aP = adjacent_sim(P); aR = adjacent_sim(R)
    rsa_pr = rsa(P, R)
    print(f'eff-rank: mlp1 position-mean {erP:.2f}  |  RoPE encoding {erR:.2f}  (RoPE dims {R.shape[1]})', flush=True)
    print(f'adjacent-position cosine sim: mlp1 {aP:.3f}  |  RoPE {aR:.3f}', flush=True)
    print(f'RSA(mlp1 position geometry, RoPE geometry) {rsa_pr:.3f}', flush=True)

    p0 = erR > 4
    pa = erP < 0.2*erR
    pb = aP > aR
    out = {'n_positions': int(P.shape[0]), 'rope_dims': int(R.shape[1]), 'eff_rank_posmean': round(erP, 3),
           'eff_rank_rope': round(erR, 3), 'adjacent_sim_posmean': round(aP, 4), 'adjacent_sim_rope': round(aR, 4),
           'rsa_posmean_rope': round(rsa_pr, 4), 'pred_0': bool(p0), 'pred_a_mlp_collapses_rope': bool(pa),
           'pred_b_smoother': bool(pb), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) MLP position rep far lower-rank than RoPE (P<0.2*R): {pa}; (b) smoother (adjacent sim higher): {pb}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
