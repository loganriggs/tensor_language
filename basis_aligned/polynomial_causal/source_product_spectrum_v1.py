"""Exact same-source key-product spectrum; no activation fitting.

Pred_a explicit small tensor Gram replay <=1e-10 relative.
Pred_b rank <=128 reaches10% coefficient Frobenius error.
Pred_c dense two-adapter replacement saves at least10% local scalars.
Scope: inside/inside key numerator only; arbitrary independent query ports.
Native denominators, position rotations, value and outside terms remain.
"""
import json
import math
import time
from pathlib import Path
import torch


def gram(a, b):
    # Orthonormal basis E_ii and (E_ij+E_ji)/sqrt(2) of Sym^2(R^d).
    d = a.shape[1]
    i, j = torch.triu_indices(d, d)
    scale = torch.where(i == j, .5, 1 / math.sqrt(2))
    g, h = a.T @ a, b.T @ b
    out = (g[i[:, None], i] * h[j[:, None], j]
           + g[i[:, None], j] * h[j[:, None], i]
           + g[j[:, None], i] * h[i[:, None], j]
           + g[j[:, None], j] * h[i[:, None], i])
    return out * scale[:, None] * scale[None, :]


def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    start = time.perf_counter()
    gen = torch.Generator().manual_seed(7131331)
    a, b = [torch.randn(7, 4, generator=gen) for _ in range(2)]
    i, j = torch.triu_indices(4, 4)
    cols = []
    for x, y in zip(i, j):
        col = torch.outer(a[:, x], b[:, y])
        if x != y:
            col = (col + torch.outer(a[:, y], b[:, x])) / math.sqrt(2)
        cols.append(col.flatten())
    explicit = torch.stack(cols, 1)
    control = float((gram(a, b) - explicit.T @ explicit).norm()
                    / (explicit.T @ explicit).norm())
    p = Path(__file__).resolve().parent
    native = torch.load(p / 'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt', weights_only=True)
    basis = native['key_basis'][1].double()
    a, b = [native[k][1].double() @ basis for k in ('k1', 'k2')]
    g = gram(a, b)
    ev = torch.linalg.eigvalsh(g).flip(0)
    assert float(ev.min()) > -1e-10 * float(ev.max())
    ev = ev.clamp_min(0)
    tail = torch.cat((ev.flip(0).cumsum(0).flip(0), torch.zeros(1))) / ev.sum()
    rank = int(torch.nonzero(tail <= .01)[0])
    # Full output feature map plus source polynomial adapter, not just rank.
    old = a.numel() + b.numel()
    output_dim, input_dim = a.shape[0] * b.shape[0], g.shape[0]
    new = rank * (input_dim + output_dim)
    result = dict(pred_a=control <= 1e-10, pred_b=rank <= 128,
                  pred_c=new <= .9 * old, control_relative_error=control,
                  input_product_dimension=input_dim, output_pair_dimension=output_dim,
                  rank_for_10_percent=rank,
                  relative_errors={str(r): float(tail[r].sqrt()) for r in (64, 128, 256, 512, 1024, 1536, 2080)},
                  local_scalars_before=old, dense_adapter_scalars_after=new,
                  minimum_nonzero_dense_adapter_scalars=input_dim + output_dim,
                  minimum_eigenvalue=float(ev.min()), maximum_eigenvalue=float(ev.max()),
                  seconds=time.perf_counter()-start, scope=__doc__)
    (p / 'SOURCE_PRODUCT_SPECTRUM_V1_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    # Countercheck: exact repeated-input Gaussian function norm, not text fitting.
    d = a.shape[1]
    i, j = torch.triu_indices(d, d)
    t = (i == j).double()
    s = math.sqrt(2)
    c = (math.sqrt(d + 2) - s) / d
    gt = g @ t
    weighted = (2*g + s*c*(torch.outer(t, gt) + torch.outer(gt, t))
                + c*c*float(t @ gt)*torch.outer(t, t))
    evg = torch.linalg.eigvalsh(weighted).flip(0).clamp_min(0)
    tailg = torch.cat((evg.flip(0).cumsum(0).flip(0), torch.zeros(1))) / evg.sum()
    ga, gb = a.T @ a, b.T @ b
    expected = ga.trace()*gb.trace() + 2*(ga @ gb).trace()
    replay = float(abs(weighted.trace()-expected)/expected)
    assert replay < 1e-10
    audit = dict(gaussian_rank_for_10_percent=int(torch.nonzero(tailg <= .01)[0]),
                 gaussian_relative_error_rank128=float(tailg[128].sqrt()),
                 fourth_moment_trace_replay=replay,
                 scope='Text-independent isotropic Gaussian source coordinates; full output key pair, no QK query normalization/value/background distribution. Dense output-rank family only.')
    (p / 'SOURCE_PRODUCT_SPECTRUM_V1_GAUSSIAN_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
