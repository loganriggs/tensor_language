"""Uniform spectral upper bounds over the frozen three-producer function span.

For Frobenius-orthonormal Q_i and ||c||=1, Q(c)^2 <= sum_i Q_i^2.
Ky Fan sums then bound rank-r captured energy for every output mixture.
These are coefficient-space bounds, not reachable-state or whole-DAG bounds.
"""
import hashlib
import json
from pathlib import Path

import torch


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    p = Path(__file__).parent
    out = p/'PARENT1_PRODUCER_MIXTURE_BOUND_V1.json'
    assert not out.exists()
    source = p/'PARENT1_MLP16_PRODUCER_V1.pt'
    saved = torch.load(source, weights_only=True, map_location='cpu')
    binding = json.loads((p/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    left, right = [weights[f'transformer.h.16.mlp.{key}.weight'].double() for key in ('Left', 'Right')]
    matrices = []
    for row in saved['folded_down']:
        raw = (left.T*row)@right
        matrix = (raw+raw.T)/2
        matrix -= matrix.trace()/len(matrix)*torch.eye(len(matrix))
        matrices.append(matrix)
    matrices = torch.stack(matrices)
    flat = matrices.flatten(1)
    gram = flat@flat.T
    energy, vectors = torch.linalg.eigh(gram)
    assert float(energy.min()/energy.max())>1e-10
    normalized = torch.einsum('gi,gab->iab', vectors/energy.sqrt()[None], matrices)
    f = normalized.flatten(1)
    gram_error = float((f@f.T-torch.eye(3)).abs().max())
    envelope = (normalized@normalized).sum(0)
    spectrum = torch.linalg.eigvalsh(envelope).flip(0).clamp_min(0)
    cumulative = spectrum.cumsum(0)
    bounds = {str(r): min(1., float(cumulative[r-1])) for r in (1,2,4,8,16,31,32,64,128,256)}
    rank90_lower = int(torch.searchsorted(cumulative, torch.tensor(.90)))+1
    generator = torch.Generator().manual_seed(5511)
    coefficients = [torch.tensor([1.,0.,0.]), torch.ones(3)/3**.5]
    coefficients += [torch.randn(3, generator=generator) for _ in range(2)]
    checks = []
    for c in coefficients:
        c = c/c.norm()
        q = torch.einsum('i,ijk->jk', c, normalized)
        values = torch.linalg.eigvalsh(q).square().sort(descending=True).values.cumsum(0)
        violations = [max(0., float(values[r-1])-bounds[str(r)]) for r in (1,2,16,31,64)]
        checks.append(dict(coefficients=c.tolist(), norm_error=abs(float(q.square().sum())-1),
                           spectral_bound_violation=max(violations),
                           rank16_capture=float(values[15])))
    error = max(gram_error, *[max(row['norm_error'], row['spectral_bound_violation']) for row in checks])
    result = dict(pred_a=error<=1e-9, pred_b=error<=1e-9 and bounds['16']<.9,
                  pred_c=error<=1e-9 and bounds['31']<.9,
                  orthonormal_function_gram_error=gram_error, mixture_checks=checks,
                  uniform_capture_upper_bounds=bounds, rank90_uniform_lower_bound=rank90_lower,
                  envelope_trace=float(spectrum.sum()), coefficient_gram_condition=float(energy.max()/energy.min()),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Uniform over the three trace-removed native MLP16 producer quadratics for the frozen parent1 input span. Rank-r signed-square approximation in coefficient Frobenius norm only; numerical FP64 bounds, no formal interval certificate. Does not constrain different readers, reachable input distributions, other program families or the whole model.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    main()
