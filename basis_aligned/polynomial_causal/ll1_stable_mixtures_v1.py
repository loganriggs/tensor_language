"""Exact signed-mixture red team of failed <=4-group coarsening.

Generalized Rayleigh modes compare the same mixtures of two fitted group banks.
A: generalized eigen/orthogonality and selected CP replay <=1e-8.
B: >=16 modes have relative function difference <=.1.
C: >=5/10 unstable anchors project >=50% energy into that old stable subspace.
This is an in-sample relation diagnostic, not an identified LL1/DAG circuit.
"""
import hashlib
import json
from pathlib import Path

import torch

from ll1_group_matching_v1 import group_inner
from ll1_joint_parent_graph_v2 import factors
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(__file__).parent
    out = root/'LL1_STABLE_MIXTURES_V1.json'
    artifact = root/'LL1_STABLE_MIXTURES_V1.pt'
    assert not out.exists() and not artifact.exists()
    paths = [root/'PROJECTED_LL1_CONVERGENCE_V3_SPECTRAL.pt',
             root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_ORIGINAL.pt']
    a = torch.load(paths[0], weights_only=True, map_location='cpu')['parts']
    b = factors(torch.load(paths[1], weights_only=True, map_location='cpu'))
    aa, bb, ab = group_inner(a, a), group_inner(b, b), group_inner(a, b)
    k = aa+bb-ab-ab.T
    eigen, basis = torch.linalg.eigh(aa)
    keep = eigen > eigen.max()*1e-12
    whitening = basis[:, keep] / eigen[keep].sqrt()
    reduced = whitening.T@k@whitening
    values, rotation = torch.linalg.eigh((reduced+reduced.T)/2)
    modes = whitening@rotation
    assert float(values.min()) >= -1e-8
    relative = values.clamp_min(0).sqrt()
    stable = relative <= .1
    orth = float((modes.T@aa@modes-torch.eye(modes.shape[1])).abs().max())
    residual = float((k@modes-aa@modes*values).norm() /
                     ((k@modes).norm()+(aa@modes*values).norm()).clamp_min(1e-30))
    selected = sorted(set([0, len(values)//2, len(values)-1] +
                          torch.where(stable)[0].tolist()[:1]))
    replay = []
    for index in selected:
        v = modes[:, index]
        old, new = cp(a[0], a[1], a[2]*v[:, None]), cp(b[0], b[1], b[2]*v[:, None])
        el, er, cross = inner(old, old), inner(new, new), inner(old, new)
        replay.append(dict(index=index, old_norm_error=abs(float(el)-1),
                           squared_drift_error=abs(float(el+er-2*cross)-float(values[index]))))
    anchors = torch.where(ab.diag()/(aa.diag()*bb.diag()).sqrt() < .8)[0]
    projection = (aa@modes[:, stable]).square().sum(1)/aa.diag()
    coverage = int((projection[anchors] >= .5).sum())
    worst = max([orth, residual] + [max(r['old_norm_error'], r['squared_drift_error']) for r in replay])
    torch.save(dict(modes=modes, squared_relative_differences=values,
                    stable_mask=stable, old_gram=aa, difference_gram=k), artifact)
    result = dict(pred_a=worst <= 1e-8, pred_b=int(stable.sum()) >= 16,
                  pred_c=coverage >= 5, retained_reference_rank=int(keep.sum()),
                  reference_gram_condition=float(eigen[keep].max()/eigen[keep].min()),
                  stable_modes=int(stable.sum()), covered_unstable_anchors=coverage,
                  relative_difference_quantiles=torch.quantile(relative, torch.tensor([0.,.25,.5,.75,1.])).tolist(),
                  relative_differences=relative.tolist(),
                  anchor_projections=[dict(group=int(i), energy_fraction=float(projection[i])) for i in anchors],
                  orthogonality_error=orth, generalized_residual=residual, replay=replay,
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  artifact=dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                  scope='Exact signed-mixture relation across two unconverged same-start fits; '
                        'mixtures generally lose rank16 LL1 form. No held-out stability or circuit claim.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
