#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: CPU only, 128 synthetic contexts, 0 model body forwards, 120 seconds.
"""Native-weight retained-contraction objective instrument, no fitting.

pred_a complete Gram/direct error replay <=1e-10 on each of four panels.
pred_b exact folded/native factored mixed numerator replay <=1e-10 per panel.
pred_c sum of the three producer contractions replays their executor <=1e-10.
Null: wrong producer/denominator contraction invalidates the objective.
Price: fixed existing 4,899,795-byte sparse program, no new learned constants;
128 synthetic raw-state contexts, 5 positions, CPU FP64, <=120 seconds.
"""
import os
import sys
import json
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(P), str(ROOT)]


def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('CPU FP64: 4x32 synthetic raw-state panels, 5 positions, no fits/body forwards; 120s')
        return
    import numpy as np
    import torch
    from head17_output_block_objective_v1 import build
    from head17_source_interface_v1 import CHECKPOINT
    from retained_contraction_error_control_v1 import components
    from extracted_circuits.three_corner_head17_interaction_v1.ports import EPS, project, from_projections, additive
    from three_group_shared_dag_v1 import execute
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    signal.alarm(120)
    start = time.perf_counter()
    target = P/'NATIVE_RETAINED_ERROR_V1_RESULT.json'
    assert not target.exists()
    sd = torch.load(CHECKPOINT, weights_only=True, map_location='cpu', mmap=True)
    tensor, ids = build()
    package = torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt', weights_only=True, map_location='cpu')
    mask = torch.from_numpy(np.unpackbits(package['mask'].numpy(), bitorder='little', count=tensor.numel()).copy()).bool()
    flat = torch.zeros(tensor.numel(), dtype=torch.float64)
    flat[mask] = package['values'].double()
    fitted = torch.einsum('op,pih,ah->oia', package['output'].double(), flat.reshape(package['shape']), package['head'].double())
    error = fitted-tensor
    writer = torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt', weights_only=True)['output_matrix'].double()
    def head(layer, name):
        return sd[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9, 128, 1152)[2].double()
    matrices = tuple(head(17, k) for k in ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v'))
    left, right, down = (sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left', 'Right', 'Down'))
    bias = sd['transformer.h.17.mlp.Down_bias'].double()
    output = sd['lm_head.weight'][ids].double()
    mixture = float(sd['transformer.h.17.attn.lamb'])
    cells = []
    with torch.no_grad():
        for seed in range(170214, 170218):
            generator = torch.Generator().manual_seed(seed)
            def rand():
                return torch.randn(32, 5, 1152, generator=generator, dtype=torch.float64)
            native, dc, dr, initial = rand(), .1*rand(), .2*rand(), rand()
            corners = (native, native+dc, native+dr)
            projections = tuple(project(x, matrices) for x in corners)
            norms = tuple(x.square().mean(-1)+EPS for x in corners)
            added_norm = norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
            first_values = (initial/(initial.square().mean(-1)+EPS).sqrt()[..., None])@head(0, 'c_v').T
            ports = tuple(from_projections(p, r, first_values, mixture) for p, r in zip(projections, norms))
            ports += (from_projections(additive(*projections), added_norm, first_values, mixture),)
            terms = components(*ports)
            producer = execute(*ports)
            producer_error = float((terms.sum(1)-producer).norm()/producer.norm())
            # This background is an explicit synthetic modeling measure, not a
            # native MLP-only state or reconstructed upstream model trajectory.
            background = native[:, -1]
            write = producer@writer.T
            z = background+write
            rms_squared = z.square().mean(-1)+EPS
            normalized = z/rms_squared.sqrt()[:, None]
            final = z+(normalized@left.T)*(normalized@right.T)@down.T+bias
            denominator = rms_squared*(final.square().mean(-1)+EPS).sqrt()
            branch = torch.einsum('oih,ni,nkh->nko', error, background, terms)/denominator[:, None, None]
            direct = torch.einsum('oih,ni,nh->no', error, background, producer)/denominator[:, None]
            gram = torch.einsum('nko,nlo->kl', branch, branch)/len(native)
            energy = float(direct.square().sum()/len(native))
            folded = torch.einsum('oih,ni,nh->no', tensor, background, producer)
            factored = ((background@left.T)*(write@right.T)+(background@right.T)*(write@left.T))@(output@down).T
            gram_error = abs(float(gram.sum())-energy)/max(energy, 1e-30)
            formula_error = float((folded-factored).norm()/factored.norm())
            cells.append(dict(seed=seed, gram=gram.tolist(), joint_error_energy=energy,
                diagonal_to_joint=float(gram.trace())/energy,
                relative_mixed_error=float(direct.norm()/(folded/denominator[:, None]).norm()),
                gram_replay_relative_error=gram_error, formula_relative_error=formula_error,
                producer_replay_relative_error=producer_error,
                pred_a=gram_error<=1e-10, pred_b=formula_error<=1e-10, pred_c=producer_error<=1e-10))
    result = dict(cells=cells, pred_a=all(c['pred_a'] for c in cells), pred_b=all(c['pred_b'] for c in cells),
        seconds=time.perf_counter()-start, checkpoint_sha256='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3',
        scope='Actual frozen weights; synthetic Gaussian raw states and unconstrained edits, not native text/circuit intervention inputs. '
              'Complete retained three-contraction producer with normalized two-QK factors and inherited values. '
              'Shared synthetic background and exact conditional MLP/final RMS reference, no softcap or altered suffix recomputation. '
              'Finite Monte Carlo instrument only: no converged population objective, fitting, behavioral or compression improvement claim.')
    result.update({'pred_a': all(c['pred_a'] for c in cells),
                   'pred_b': all(c['pred_b'] for c in cells),
                   'pred_c': all(c['pred_c'] for c in cells)})
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))
    assert result['pred_a'] and result['pred_b'] and result['pred_c']


if __name__ == '__main__':
    main()
