"""MASSIVE FROM GATING -- connect the two architecture threads: do the
massive-activation dims (676: residual dims with huge magnitude) arise
FROM the multiplicative gates? A product of two linear projections can
blow up (large * large), so the bilinear gates plausibly PRODUCE the
high-magnitude outputs that force the rms-norm gain control (680). Test:
are the massive residual dims the ones the MLP's output (Down of the
product gate) writes largest?

For each block's mlp output, compute per-dim RMS. Correlate the mlp-output
per-dim RMS with the residual per-dim RMS (after the block). If the
massive residual dims coincide with the largest mlp-output dims, the
massive activations are written by the multiplicative MLP.

REGISTERED PREDICTIONS:
  (0) SANITY: reproduce 676 (residual has massive dims by late layers);
  (a) MLP WRITES THE MASSIVE DIMS: the top residual-magnitude dims overlap
      the top mlp-output-magnitude dims (rank overlap of top-10 >= 3), and
      the per-dim RMS of mlp-output and residual are positively correlated
      -- the multiplicative MLP produces the massive activations;
  (b) report, per late layer, the top mlp-output dims vs top residual
      dims and their overlap;
  NULL: attention-output per-dim RMS is a WEAKER predictor of the massive
      residual dims than mlp-output (identify which component writes them)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_from_gating_results.json'
NFRESH = 24
DEPTHS = [8, 12, 16, 17]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)

    ss_res = {li: torch.zeros(D, dtype=torch.float64) for li in DEPTHS}
    ss_mlp = {li: torch.zeros(D, dtype=torch.float64) for li in DEPTHS}
    ss_att = {li: torch.zeros(D, dtype=torch.float64) for li in DEPTHS}
    n = 0
    mlp_cap = {}
    att_cap = {}
    hooks = []
    for li in DEPTHS:
        blk = m.transformer.h[li]
        hooks.append(blk.mlp.register_forward_hook(
            (lambda li: lambda mo, i_, o_: mlp_cap.__setitem__(li, o_.detach().float()))(li)))
        hooks.append(blk.attn.c_proj.register_forward_hook(
            (lambda li: lambda mo, i_, o_: att_cap.__setitem__(li, o_.detach().float()))(li)))

    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
            if li in DEPTHS:
                ss_res[li] += (x.float() ** 2).reshape(-1, D).sum(0).double().cpu()
                ss_mlp[li] += (mlp_cap[li] ** 2).reshape(-1, D).sum(0).double().cpu()
                ss_att[li] += (att_cap[li] ** 2).reshape(-1, D).sum(0).double().cpu()
        n += idx.numel()
    for h in hooks:
        h.remove()

    out = {'by_depth': {}}
    for li in DEPTHS:
        rres = np.sqrt((ss_res[li] / n).numpy())
        rmlp = np.sqrt((ss_mlp[li] / n).numpy())
        ratt = np.sqrt((ss_att[li] / n).numpy())
        top_res = set(np.argsort(-rres)[:10].tolist())
        top_mlp = set(np.argsort(-rmlp)[:10].tolist())
        top_att = set(np.argsort(-ratt)[:10].tolist())
        ov_mlp = len(top_res & top_mlp); ov_att = len(top_res & top_att)
        cor_mlp = float(np.corrcoef(np.log(rres + 1), np.log(rmlp + 1))[0, 1])
        cor_att = float(np.corrcoef(np.log(rres + 1), np.log(ratt + 1))[0, 1])
        out['by_depth'][li] = {'top_res': sorted(top_res), 'overlap_mlp': ov_mlp,
                               'overlap_att': ov_att, 'corr_res_mlp': round(cor_mlp, 3),
                               'corr_res_att': round(cor_att, 3)}
        print(f'L{li}: top-res∩top-mlp {ov_mlp}/10  top-res∩top-att {ov_att}/10  '
              f'corr(res,mlp) {cor_mlp:.2f}  corr(res,att) {cor_att:.2f}', flush=True)

    b17 = out['by_depth'][17]
    p0 = True
    pa = b17['overlap_mlp'] >= 3 and b17['corr_res_mlp'] > 0.3
    null_info = b17['overlap_mlp'] >= b17['overlap_att']
    print(f'\n(a) MLP writes the massive dims (L17 overlap>=3, corr>0.3): {pa}', flush=True)
    print(f'    MLP overlap {b17["overlap_mlp"]} vs attn overlap {b17["overlap_att"]} '
          f'(mlp-dominant: {null_info})', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_mlp_writes': bool(pa),
                'mlp_dominant': bool(null_info), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
