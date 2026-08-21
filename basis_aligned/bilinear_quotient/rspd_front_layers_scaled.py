"""RSPD FRONT LAYERS SCALED (user feedback: careful with data volume; scale
on GPU; map the first few layers ENTIRELY). Two things at once:

 (1) DATA-ROBUSTNESS of the r80 claim -- the centerpiece. The 694-697
     numbers used only 3072 tokens. Here we refit mlp0.Down's A-SVD at
     N in {3k, 6k, 12k, 24k} tokens (all on GPU) and price each r80 on a
     FIXED held-out eval set. If r80 barely moves, the low-rank claim is
     data-robust; if it climbs with N, the earlier claim was a small-
     sample artifact (which we would then state plainly).

 (2) SYSTEMATIC FRONT MAP -- every core component of blocks 0,1,2 (each
     block's attn.c_proj AND mlp.Down = 6 components), one CE-priced A-SVD
     r80 each at a common N=12k tokens, so the first three layers are
     characterized together rather than one-off.

Everything runs on GPU (weights + activations on DEV; A-SVD pinv on GPU).
CE is priced by substituting the rank-r surrogate into the live model and
running the full forward (so downstream effects of a front component are
included).

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank A-SVD reproduces baseline CE for each component
      (|d| < 0.01);
  (a) DATA-ROBUST r80 (the key test): mlp0.Down's r80 stays within a factor
      of 2 across N=3k..24k (i.e. does NOT keep climbing with data) -- the
      low-rank core is a real property, not a small-sample artifact. [If it
      FAILS -- r80 grows with N -- report that honestly as a correction.]
  (b) FRONT MAP: report r80 + benefit for all 6 components of blocks 0-2;
      register the expectation that mlp Down layers are lower-rank than
      attention c_proj (MLPs = few functional writers; attention = routing);
  NULL: random rank-r projection recovers ~0/negative (r80=never) for each."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace/rspd')
from rspd.asvd import generate_lowrank_approximation
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_front_layers_scaled_results.json'
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_EVAL_ROWS = 48          # ~12k tokens held-out for CE pricing
N_MAP = 48                # ~12k tokens for the 6-component fit
SCALE_ROWS = [12, 24, 48, 96]   # ~3k,6k,12k,24k tokens for the robustness sweep


def submod(block_i, kind):
    blk = m.transformer.h[block_i]
    return (blk.attn.c_proj if kind == 'attn' else blk.mlp.Down)


@torch.no_grad()
def forward_ce(fresh, rows_slice):
    ce_s = 0.0; n = 0
    for i in range(0, rows_slice, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        ce_s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1),
                                 reduction='mean')) * idx.shape[0]
        n += idx.shape[0]
    return ce_s / n


@torch.no_grad()
def ce_with_weight(mod, Wsub, eval_rows, eval_n):
    orig = mod.weight.data
    if Wsub == 'ablate':
        mod.weight.data = torch.zeros_like(orig)
    elif Wsub is not None:
        mod.weight.data = Wsub.to(orig.dtype).to(orig.device)
    ce = forward_ce(eval_rows, eval_n)
    mod.weight.data = orig
    return ce


@torch.no_grad()
def capture_input(mod, cap_rows, cap_n, in_dim):
    cap = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, cap_n, 4):
        bb = cap_rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0)          # on DEV


@torch.no_grad()
def asvd_r80(mod, X, W, eval_rows, eval_n, ce_full, ce_ablate, ranks=RANKS):
    """Fit A-SVD on (W, X) [GPU], price each rank-r by real CE, return the
    recovered-curve, r80, and random-projection null r80."""
    benefit = ce_ablate - ce_full
    A_fac, B_fac = generate_lowrank_approximation(W, X, target=X @ W.T)
    maxr = A_fac.shape[1]
    g = torch.Generator(device='cpu').manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(W.shape[0], W.shape[0], generator=g))
    Q = Q.to(DEV)
    rec = {}; rec_rand = {}
    for r in ranks:
        if r > maxr:
            break
        W_r = A_fac[:, :r] @ B_fac[:r, :]
        rec[r] = round(float((ce_ablate - ce_with_weight(mod, W_r, eval_rows, eval_n)) / benefit), 4)
        Qr = Q[:, :r]
        Wr_rand = Qr @ (Qr.T @ W)
        rec_rand[r] = round(float((ce_ablate - ce_with_weight(mod, Wr_rand, eval_rows, eval_n)) / benefit), 4)
    def r80(tbl):
        for r in ranks:
            if r in tbl and tbl[r] >= 0.80:
                return r
        return None
    W_full = A_fac @ B_fac
    ce_fr = ce_with_weight(mod, W_full, eval_rows, eval_n)
    return rec, rec_rand, r80(rec), r80(rec_rand), round(ce_fr, 4)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(96 + N_EVAL_ROWS)     # fit pool + disjoint eval
    ev = allrows[96:96 + N_EVAL_ROWS]               # held-out eval
    ce_base = forward_ce(ev, N_EVAL_ROWS)
    print(f'baseline CE (eval {N_EVAL_ROWS} rows): {ce_base:.4f}', flush=True)

    out = {'baseline_ce': round(ce_base, 4), 'scaling': {}, 'front_map': {}}

    # (1) DATA-ROBUSTNESS SWEEP on mlp0.Down
    mod0 = submod(0, 'mlp'); W0 = mod0.weight.data.float()
    ce_full0 = ce_base
    ce_ablate0 = ce_with_weight(mod0, 'ablate', ev, N_EVAL_ROWS)
    print(f'\n=== data-scaling sweep: mlp0.Down (benefit {ce_ablate0-ce_full0:.3f}) ===',
          flush=True)
    for nr in SCALE_ROWS:
        X = capture_input(mod0, allrows[:nr], nr, 4608)
        rec, rec_rand, r80a, r80r, ce_fr = asvd_r80(mod0, X, W0, ev, N_EVAL_ROWS,
                                                     ce_full0, ce_ablate0)
        ntok = nr * 256
        out['scaling'][ntok] = {'r80': r80a, 'r80_random': r80r, 'recovered': rec,
                                'ce_fullrank': ce_fr}
        print(f'N={ntok:6d} tok: r80={r80a}  rand_r80={r80r}  '
              f'rec@4={rec.get(4)} rec@8={rec.get(8)} rec@32={rec.get(32)}', flush=True)
        del X; torch.cuda.empty_cache()

    r80s = [out['scaling'][k]['r80'] for k in out['scaling'] if out['scaling'][k]['r80']]
    robust = len(r80s) >= 2 and (max(r80s) <= 2 * min(r80s))
    print(f'\n(a) r80 across N = {r80s} -> data-robust (<=2x spread): {robust}', flush=True)

    # (2) FRONT MAP: 6 components of blocks 0-2 at N=12k
    print(f'\n=== front map (blocks 0-2, N={N_MAP*256} tok) ===', flush=True)
    for bi in [0, 1, 2]:
        for kind, in_dim in [('attn', 1152), ('mlp', 4608)]:
            mod = submod(bi, kind); W = mod.weight.data.float()
            ce_full = ce_base
            ce_abl = ce_with_weight(mod, 'ablate', ev, N_EVAL_ROWS)
            X = capture_input(mod, allrows[:N_MAP], N_MAP, in_dim)
            rec, rec_rand, r80a, r80r, ce_fr = asvd_r80(mod, X, W, ev, N_EVAL_ROWS,
                                                        ce_full, ce_abl)
            key = f'block{bi}.{kind}'
            out['front_map'][key] = {'benefit': round(ce_abl - ce_full, 4), 'r80': r80a,
                                     'r80_random': r80r, 'recovered': rec,
                                     'ce_fullrank': ce_fr, 'ce_ablate': round(ce_abl, 4)}
            print(f'{key:14s}: benefit {ce_abl-ce_full:5.3f}  r80={r80a}  '
                  f'rand_r80={r80r}  rec@8={rec.get(8)}', flush=True)
            del X; torch.cuda.empty_cache()

    out['data_robust'] = bool(robust)
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
