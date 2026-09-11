"""Native MLP16 quadratic producers for frozen parent1's three input ports.

Use orthonormal coordinates for their span, retaining a map back to the actual
ports. The MLP17 RMS divisor and intervening attention/residual background
remain explicit; this is a conditional producer, not a text-to-answer circuit.
"""
import hashlib
import json
from pathlib import Path

import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    p = Path(__file__).parent
    out = p/'PARENT1_MLP16_PRODUCER_V1.json'
    assert not out.exists()
    source = p/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    saved = torch.load(source, weights_only=True, map_location='cpu')
    node = saved['nodes'][1]
    frame = torch.cat((node['reader'][None], node['partners'].T), 0).double()
    basis, _ = torch.linalg.qr(frame.T, mode='reduced')
    adapter = frame @ basis
    binding = json.loads((p/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint = next(name for name in binding if name.endswith('/pytorch_model.bin'))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    left, right, down = [weights[f'transformer.h.16.mlp.{key}.weight'].double() for key in ('Left', 'Right', 'Down')]
    bias = weights['transformer.h.16.mlp.Down_bias'].double()
    beta = weights['transformer.h.17.lambdas'].double()[0]
    folded = beta * basis.T @ down
    quadratics = []
    for row in folded:
        raw = (left.T*row) @ right
        quadratics.append((raw+raw.T)/2)
    quadratics = torch.stack(quadratics)
    folded_bias = beta * (basis.T @ bias)
    dimension = left.shape[1]
    radial = quadratics.diagonal(dim1=1, dim2=2).sum(1)/dimension
    centered = quadratics-radial[:, None, None]*torch.eye(dimension)
    def output_modes(q):
        flat = q.flatten(1)
        energy, vectors = torch.linalg.eigh(flat @ flat.T)
        return energy.flip(0).clamp_min(0), vectors.flip(1)
    raw_energy, _ = output_modes(quadratics)
    centered_energy, output_basis = output_modes(centered)
    modes = torch.einsum('gj,gab->jab', output_basis, centered)
    spectra, compact = [], []
    for j, matrix in enumerate(modes):
        values, vectors = torch.linalg.eigh(matrix)
        order = values.abs().argsort(descending=True)
        cumulative = values[order].square().cumsum(0)/values.square().sum()
        one_product = (values.clamp_min(0).max().square()+values.clamp_max(0).min().square())/values.square().sum()
        spectra.append(dict(mode=j, fraction_of_trace_removed_energy=float(centered_energy[j]/centered_energy.sum()),
                            best_one_real_product_capture=float(one_product),
                            square_capture={str(k):float(cumulative[k-1]) for k in (1,2,4,8,16,64,256)},
                            rank90=int(torch.searchsorted(cumulative, torch.tensor(.90)))+1,
                            rank99=int(torch.searchsorted(cumulative, torch.tensor(.99)))+1))
        compact.append(dict(readers=vectors[:, order[:16]].T, weights=values[order[:16]]))
    torch.manual_seed(5501)
    raw16 = torch.randn(19, dimension)
    epsilon = torch.finfo(torch.float32).eps
    rho16 = (raw16.square().mean(1)+epsilon).sqrt()
    x = raw16/rho16[:, None]
    products = (x @ left.T)*(x @ right.T)
    direct_producer = beta * (products @ down.T+bias)
    calculated = torch.einsum('bi,gij,bj->bg', x, quadratics, x)+folded_bias
    reference = direct_producer @ basis
    producer_error = float((calculated-reference).norm()/reference.norm())
    radial_rebuild = (torch.einsum('bi,gij,bj->bg', x, centered, x)+
                      x.square().sum(1)[:, None]*radial+folded_bias)
    radial_error = float((radial_rebuild-reference).norm()/reference.norm())
    background = torch.randn_like(raw16)
    raw17 = background+direct_producer
    rho17 = (raw17.square().mean(1)+epsilon).sqrt()
    normalized17 = raw17/rho17[:, None]
    original_ports = normalized17 @ frame.T
    compiled_ports = (background @ frame.T+calculated @ adapter.T)/rho17[:, None]
    port_error = float((compiled_ports-original_ports).norm()/original_ports.norm())
    writers = torch.linalg.solve_triangular(saved['output_whitener'].double(), node['writers'].double(), upper=True)
    original = (original_ports[:, :1]*original_ports[:, 1:]) @ writers.T
    rebuilt = (compiled_ports[:, :1]*compiled_ports[:, 1:]) @ writers.T
    branch_error = float((rebuilt-original).norm()/original.norm())
    errors = dict(producer=producer_error, radial_retention=radial_error, ports=port_error,
                  whole_node_branch_replay=branch_error,
                  orthonormal_span=float((basis.T@basis-torch.eye(3)).abs().max()),
                  frame_reconstruction=float((adapter@basis.T-frame).norm()/frame.norm()))
    a = all(value<=1e-9 for value in errors.values())
    raw_capture = float(raw_energy[0]/raw_energy.sum())
    centered_capture = float(centered_energy[0]/centered_energy.sum())
    artifact = p/'PARENT1_MLP16_PRODUCER_V1.pt'
    torch.save(dict(input_span=basis, actual_port_adapter=adapter, folded_down=folded,
                    folded_bias=folded_bias, radial_coefficients=radial,
                    centered_output_modes=output_basis, top16_square_modes=compact,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    scope='Exact folded Down plus native Left/Right and background are required. Top16 modes are approximations, not adopted replacements.'), artifact)
    result = dict(pred_a=a, pred_b=a and min(raw_capture, centered_capture)>=.9,
                  pred_c=a and all(row['square_capture']['16']>=.9 for row in spectra),
                  errors=errors, block17_reentry_coefficient=float(beta),
                  raw_output_rank1_capture=raw_capture, trace_removed_output_rank1_capture=centered_capture,
                  raw_output_energy_fractions=(raw_energy/raw_energy.sum()).tolist(),
                  trace_removed_output_energy_fractions=(centered_energy/centered_energy.sum()).tolist(),
                  radial_energy_fraction=float((dimension*radial.square().sum())/quadratics.square().sum()),
                  spectra=spectra, artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  scope='Frozen three-port producer subspace of native MLP16. Conditional on intervening attention/residual background and actual RMS norms. Spectral statements are coefficient-space restrictions, not natural-input fidelity, independent extraction, OOD or semantic circuit identification.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    main()
