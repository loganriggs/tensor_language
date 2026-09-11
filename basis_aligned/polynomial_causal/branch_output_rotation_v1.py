"""Exact 2D varimax within a frozen shared-reader node; no data fitting.

All 50304 unembedding rows, uncentered. Orthogonal writers have unit full-U
norms; their old norms move into partner readers before the common rotation.
Registered A numeric identities <=1e-9; B fourth moment gains >=10%;
C overlap mass falls >=25%. These are structural, not behavioral, bars.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    target = root / 'BRANCH_OUTPUT_ROTATION_V1.json'
    assert not target.exists()
    source = root / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    saved = torch.load(source, weights_only=True, map_location='cpu')
    node = saved['nodes'][1]
    c, p = node['writers'].double(), node['partners'].double()
    assert c.shape == p.shape == (1152, 2)
    scales = c.norm(dim=0)
    o, k = c / scales, p * scales
    binding = json.loads((root / 'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    physical = torch.linalg.solve_triangular(saved['output_whitener'].double(), o, upper=True)
    loadings = weights['lm_head.weight'].double() @ physical
    a, b = loadings.unbind(1)
    const = .75 * (a.square() + b.square()).square().sum()
    alpha = .25 * (a.pow(4) - 6*a.square()*b.square() + b.pow(4)).sum()
    beta = (a*b*(a.square()-b.square())).sum()
    theta = torch.atan2(beta, alpha) / 4
    co, si = theta.cos(), theta.sin()
    rotation = torch.stack((torch.stack((co, -si)), torch.stack((si, co))))
    rotated = loadings @ rotation
    new_c, new_p = o @ rotation, k @ rotation
    old_m, new_m = c @ p.T, new_c @ new_p.T
    score0, score1 = loadings.pow(4).sum(), rotated.pow(4).sum()
    expected = const + torch.hypot(alpha, beta)
    identity = torch.eye(2)
    errors = {
        'writer_orthogonality': float((o.T @ o - identity).abs().max()),
        'full_U_gram': float((loadings.T @ loadings - identity).abs().max()),
        'rotation_orthogonality': float((rotation.T @ rotation - identity).abs().max()),
        'analytic_global_optimum_relative': float((score1-expected).abs()/expected),
        'analytic_original_relative': float((score0-const-alpha).abs()/score0),
        'stationarity_relative': float((-4*alpha*torch.sin(4*theta)+4*beta*torch.cos(4*theta)).abs()/expected),
        'full_node_matrix_relative': float((new_m-old_m).norm()/old_m.norm()),
    }
    # Independent direct function replay on random inputs, no model data.
    x = torch.randn(64, 1152, generator=torch.Generator().manual_seed(5101))
    u = node['reader'].double()
    old = (x @ u)[:, None] * ((x @ p) @ c.T)
    new = (x @ u)[:, None] * ((x @ new_p) @ new_c.T)
    errors['function_replay'] = float((new-old).norm()/old.norm())
    overlap0 = loadings.square().min(1).values.sum()
    overlap1 = rotated.square().min(1).values.sum()
    gain = float(score1/score0-1)
    reduction = float(1-overlap1/overlap0)
    tokenizer_path = Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    def annotate(f):
        result = []
        for j in range(2):
            col = f[:50257, j]
            entry = {}
            for name, sign in [('positive', 1), ('negative', -1)]:
                ids = (col*sign).topk(20).indices.tolist()
                entry[name] = [dict(id=i, token=tokenizer.decode([i]), loading=float(col[i])) for i in ids]
            result.append(entry)
        return result
    artifact = root / 'BRANCH_OUTPUT_ROTATION_V1.pt'
    torch.save(dict(reader=u, writers=new_c, partners=new_p, rotation=rotation,
                    output_whitener=saved['output_whitener'], source_node=1), artifact)
    valid = all(math.isfinite(v) and v <= 1e-9 for v in errors.values())
    result = dict(pred_a=valid, pred_b=valid and gain >= .10,
                  pred_c=valid and reduction >= .25,
                  angle_radians=float(theta), angle_degrees=float(theta*180/math.pi),
                  errors=errors, fourth_moment_before=float(score0),
                  fourth_moment_after=float(score1), fourth_moment_relative_gain=gain,
                  overlap_before=float(overlap0), overlap_after=float(overlap1),
                  overlap_relative_reduction=reduction,
                  original=annotate(loadings), rotated=annotate(rotated),
                  price=dict(reader_coefficients=1152, writer_coefficients=2304,
                             partner_coefficients=2304, variable_products=2,
                             additional_execution_cost=0,
                             note='Rotation absorbed into existing coefficients; common native background still required.'),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  scope='Exact global 2D orthogonal rotation optimum only. Summed node function unchanged; individual branch deletions change. No data fitting or circuit identification.')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('original', 'rotated')}, indent=2))


if __name__ == '__main__':
    main()
