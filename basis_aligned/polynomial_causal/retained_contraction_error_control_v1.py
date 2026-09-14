"""Synthetic control for a complete retained-producer mixed-error objective.

Predictions frozen before execution: producer replay and full Gram error replay
relative errors <= 1e-10; cancelling branches have zero joint energy but positive
diagonal energy. Uses the existing normalized projected-port implementation.
No learned weights, text-derived objective, fit, or native behavioral claim.
"""
import json
from pathlib import Path

import torch

from extracted_circuits.three_corner_head17_interaction_v1.ports import (
    EPS, additive, from_projections, project,
)
from three_group_shared_dag_v1 import execute


def components(native, child, remainder, added):
    a, b, v = native
    ac, bc, vc = (x-y for x, y in zip(child, native))
    ar, br, vr = (x-y for x, y in zip(remainder, native))
    abar, bbar, vbar = a+ac+ar, b+bc+br, v+vc+vr
    ea, eb = added[0]-abar, added[1]-bbar
    gates = (ar*bbar+(a+ac)*br, ac*bbar+(a+ar)*bc,
             ea*bbar+abar*eb)
    return torch.stack([
        torch.einsum('ns,nsh->nh', gate, value)
        for gate, value in zip(gates, (vc, vr, vbar))
    ], dim=1)


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(170214)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    # Raw-state corners share their base, projections and inherited values.
    batch, sources, residual, head, outputs = 24, 5, 16, 8, 4
    native = rand(batch, sources, residual)
    dc, dr = .1*rand(*native.shape), .2*rand(*native.shape)
    raw = (native, native+dc, native+dr)
    matrices = tuple(rand(head, residual)/residual**.5 for _ in range(5))
    projections = tuple(project(x, matrices) for x in raw)
    norms = tuple(x.square().mean(-1)+EPS for x in raw)
    cross = (dc*dr).mean(-1)
    added_norm = norms[1]+norms[2]-norms[0]+2*cross
    first = rand(batch, sources, head)
    ports = tuple(from_projections(p, r, first, .3)
                  for p, r in zip(projections, norms))
    ports += (from_projections(additive(*projections), added_norm, first, .3),)
    terms = components(*ports)
    producer = execute(*ports)
    producer_error = float((terms.sum(1)-producer).norm()/producer.norm())
    # A single downstream operator error acts on the shared background and
    # the sum of all three producer contractions. Supply identical denominators
    # to both computations; this is a conditional raw-logit objective.
    background = native[:, -1]
    writer = rand(residual, head)/head**.5
    state = background+producer@writer.T
    input_rms_squared = state.square().mean(-1)+EPS
    supplied_final_rms = (background.square().mean(-1)+EPS).sqrt()
    denominator = input_rms_squared*supplied_final_rms
    error_operator = rand(outputs, residual, head)
    branch_errors = torch.einsum('oih,ni,nkh->nko', error_operator,
                                 background, terms)/denominator[:, None, None]
    direct = torch.einsum('oih,ni,nh->no', error_operator, background,
                          producer)/denominator[:, None]
    gram = torch.einsum('nko,nlo->kl', branch_errors, branch_errors)/batch
    joint = float(direct.square().sum()/batch)
    replay_error = abs(float(gram.sum())-joint)/joint
    diagonal = float(gram.trace())
    # Discriminating witness: dropping cross terms changes the objective even
    # when the summed function is exactly zero.
    witness = rand(batch, outputs)
    cancel = torch.stack((witness, -witness, torch.zeros_like(witness)), 1)
    witness_gram = torch.einsum('nko,nlo->kl', cancel, cancel)/batch
    cancellation_pass = bool(cancel.sum(1).square().sum() == 0
                             and witness_gram.trace() > 0)
    result = dict(
        producer_replay_relative_error=producer_error,
        complete_gram_replay_relative_error=replay_error,
        joint_squared_error=joint, diagonal_only_squared_error=diagonal,
        cross_term_energy=float(gram.sum()-gram.trace()),
        gram=gram.tolist(), cancellation_witness_pass=cancellation_pass,
        pred_a=producer_error <= 1e-10, pred_b=replay_error <= 1e-10,
        pred_c=cancellation_pass,
        scope='Synthetic normalized shared-corner producer, including both QK '
              'factors and inherited values. Conditional mixed numerator/raw '
              'logits with supplied final RMS. No native weights, downstream '
              'softcap, model extraction, fitting or behavioral evidence.',
    )
    assert all(result[k] for k in ('pred_a', 'pred_b', 'pred_c'))
    return result


if __name__ == '__main__':
    result = run()
    target = Path(__file__).with_name('RETAINED_CONTRACTION_ERROR_CONTROL_V1_RESULT.json')
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))
