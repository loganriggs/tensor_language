"""ENCODER vs DECODER RANK (user's conceptual deconfusion). The bilinear MLP
is out = Down[(Left.x) (.) (Right.x)] + b: an ENCODER (Left/Right -- which
features it detects from the residual) feeding a DECODER (Down -- what it
writes). All prior rank work measured the DECODER functional rank (mlp1=128,
mlp0=8). QUESTION: is a high-rank layer's rank in the ENCODER (it genuinely
detects many distinct features) or the DECODER (few features spread across
many outputs -- "computing the same thing, recombined")?

Measure per layer (64k tokens), all as EFFECTIVE RANK = participation ratio
(sum sigma)^2 / sum sigma^2 (a smooth rank of the singular spectrum):
  input_er   : residual input x over tokens (how many residual dirs active);
  read_er    : encoder weight [Left; Right] singular spectrum (how many
               residual dirs the gates read -- ENCODER capacity);
  gate_er    : gate activation (Left.x)(.)(Right.x) over tokens (how many
               distinct FEATURES the encoder actually produces);
  out_er     : MLP output over tokens (DECODER output variance spread);
  decoder_r80: the CE-functional decoder rank (recomputed, fast A-SVD).
Compare low-rank (mlp0,16,17) vs high-rank (mlp1,2). If high-decoder-rank
layers ALSO have high gate_er/read_er -> the diversity is in the ENCODER
(they detect more features). If gate_er is similar across layers but
decoder_r80 differs -> the difference is DECODER recombination.

REGISTERED PREDICTIONS:
  (0) SANITY: effective ranks computable; decoder_r80 matches 713;
  (a) DECONFUSION (report, no strong prior): give encoder (read_er, gate_er)
      vs decoder (out_er, r80) per layer. Success = a clear statement of
      whether mlp1's excess rank over mlp0 lives in the encoder (gate_er
      ratio ~ r80 ratio) or the decoder (gate_er similar, r80 differs);
  (b) also report per-layer the correlation structure: is the gate feature
      set of mlp1 genuinely richer (higher gate_er) than mlp0's?
  NULL: n/a (measurement) -- but a random-weight MLP's gate_er ~ full
      (participation ratio near dim) for calibration."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'encoder_vs_decoder_rank_results.json'
NFIT = 128; NEVAL = 96
LAYERS = [0, 1, 2, 4, 15, 16, 17]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def eff_rank_from_sv(sv):
    sv = sv[sv > 0]
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def eff_rank_cov(Xmat):
    # participation ratio of the covariance spectrum of rows (centered)
    Xc = Xmat - Xmat.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(Xc.float())
    return eff_rank_from_sv(sv ** 2)     # eigenvalues of cov ~ sv^2


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def forward_ce(rows, n, mod=None, W=None):
    orig = None
    if mod is not None:
        orig = mod.weight.data
        mod.weight.data = (torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype)) if W is not None else orig
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if orig is not None: mod.weight.data = orig
    return s/nn


@torch.no_grad()
def capture_layer(rows, n, li):
    """return residual input x, gate, output, gate-input for layer li."""
    xin = []; gate = []; out = []
    blk = m.transformer.h[li]
    hx = blk.mlp.register_forward_pre_hook(lambda mo, inp: xin.append(inp[0].detach().float().reshape(-1, D)))
    hg = blk.mlp.Down.register_forward_pre_hook(lambda mo, inp: gate.append(inp[0].detach().float().reshape(-1, HID)))
    ho = blk.mlp.register_forward_hook(lambda mo, i_, o_: out.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for l2, b2 in enumerate(m.transformer.h): x, v1 = b2(x, v1, x0)
    hx.remove(); hg.remove(); ho.remove()
    return torch.cat(xin,0), torch.cat(gate,0), torch.cat(out,0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    ce_full = forward_ce(ev, NEVAL)

    res = {}
    for li in LAYERS:
        blk = m.transformer.h[li]
        # encoder read effective rank (weight)
        stacked = torch.cat([blk.mlp.Left.weight.data.float(), blk.mlp.Right.weight.data.float()], 0).to(DEV)
        read_er = eff_rank_from_sv(torch.linalg.svdvals(stacked) ** 2)
        # activations
        xin, gate, out = capture_layer(fit, NFIT, li)
        # subsample rows for the big SVDs (cap tokens for svdvals speed)
        cap = 8000
        input_er = eff_rank_cov(xin[:cap]); gate_er = eff_rank_cov(gate[:cap]); out_er = eff_rank_cov(out[:cap])
        # decoder functional r80
        W = blk.mlp.Down.weight.data.float().to(DEV)
        A, B = asvd_fast(W, gate)
        ce_abl = forward_ce(ev, NEVAL, blk.mlp.Down, 'ablate'); ben = ce_abl - ce_full
        r80 = RANKS[-1]
        for r in RANKS:
            if (ce_abl - forward_ce(ev, NEVAL, blk.mlp.Down, A[:, :r] @ B[:r, :]))/max(ben,1e-6) >= 0.80:
                r80 = r; break
        res[li] = {'input_er': round(input_er,1), 'read_er': round(read_er,1),
                   'gate_er': round(gate_er,1), 'out_er': round(out_er,1),
                   'decoder_r80': int(r80), 'benefit': round(float(ben),3)}
        print(f'mlp{li:2d}: input_er {input_er:5.1f}  read_er {read_er:5.1f}  '
              f'gate_er {gate_er:6.1f}  out_er {out_er:5.1f}  decoder_r80 {r80:3d}  ben {ben:.3f}', flush=True)

    # deconfusion: ratio of high vs low layer
    hi, lo = res[1], res[0]
    print(f'\nmlp1/mlp0 ratios: gate_er {hi["gate_er"]/lo["gate_er"]:.2f}  '
          f'read_er {hi["read_er"]/lo["read_er"]:.2f}  '
          f'decoder_r80 {hi["decoder_r80"]/lo["decoder_r80"]:.2f}', flush=True)
    print('Interpretation: if gate_er ratio ~ decoder_r80 ratio -> encoder-driven; '
          'if gate_er ratio ~1 while r80 ratio >>1 -> decoder-recombination.', flush=True)

    out = {'per_layer': res, 'baseline_ce': round(ce_full,4), 'n_tokens': NFIT*256,
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
