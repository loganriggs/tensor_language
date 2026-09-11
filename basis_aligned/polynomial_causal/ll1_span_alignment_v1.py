"""Independent signed mixtures: principal angles of two fitted group spans.

A: whitening, SVD and selected CP projection replay <=1e-8.
B: >=16 directions with best projection relative error <=.1.
C: >=5/10 unstable old anchors half covered by the stable old subspace.
Exact linear alignment diagnostic on two unconverged same-start fits.
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_group_matching_v1 import group_inner
from ll1_joint_parent_graph_v2 import factors
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def whiten(gram):
    values, vectors = torch.linalg.eigh(gram)
    keep = values > values.max()*1e-12
    return vectors[:, keep]/values[keep].sqrt()


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(__file__).parent
    out, artifact = root/'LL1_SPAN_ALIGNMENT_V1.json', root/'LL1_SPAN_ALIGNMENT_V1.pt'
    assert not out.exists() and not artifact.exists()
    paths = [root/'PROJECTED_LL1_CONVERGENCE_V3_SPECTRAL.pt',
             root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_ORIGINAL.pt']
    old = torch.load(paths[0], weights_only=True, map_location='cpu')['parts']
    new = factors(torch.load(paths[1], weights_only=True, map_location='cpu'))
    aa, bb, ab = group_inner(old, old), group_inner(new, new), group_inner(old, new)
    wa, wb = whiten(aa), whiten(bb)
    cross = wa.T@ab@wb
    u, sigma, vh = torch.linalg.svd(cross)
    ma, mb = wa@u, wb@vh.T
    assert float(sigma.max()) <= 1+1e-8
    relative = (1-sigma.square()).clamp_min(0).sqrt()
    stable = relative <= .1
    numerical = [float((ma.T@aa@ma-torch.eye(ma.shape[1])).abs().max()),
                 float((mb.T@bb@mb-torch.eye(mb.shape[1])).abs().max()),
                 float((ma.T@ab@mb-torch.diag(sigma)).abs().max())]
    replay = []
    for i in (0, 31, 63):
        a = cp(old[0], old[1], old[2]*ma[:, i, None])
        b = cp(new[0], new[1], new[2]*mb[:, i, None]*sigma[i])
        drift = float(inner(a, a)+inner(b, b)-2*inner(a, b))
        error = abs(drift-float(relative[i].square()))
        numerical.append(error)
        replay.append(dict(mode=i, predicted_squared_projection_error=float(relative[i].square()),
                           exact_cp_squared_projection_error=drift, error=error))
    anchors = torch.where(ab.diag()/(aa.diag()*bb.diag()).sqrt() < .8)[0]
    projections = (aa@ma[:, stable]).square().sum(1)/aa.diag()
    covered = int((projections[anchors] >= .5).sum())
    torch.save(dict(old_modes=ma, new_modes=mb, principal_cosines=sigma,
                    stable_mask=stable), artifact)
    result = dict(pred_a=max(numerical) <= 1e-8, pred_b=int(stable.sum()) >= 16,
                  pred_c=covered >= 5, stable_directions=int(stable.sum()),
                  covered_unstable_anchors=covered, maximum_replay=max(numerical),
                  projection_relative_errors=relative.tolist(),
                  anchor_projections=[dict(group=int(i), energy_fraction=float(projections[i])) for i in anchors],
                  replay=replay,
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  artifact=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                  scope='Basis-invariant optimal alignment of existing64-dimensional function spans. '
                        'Not optimization over input readers, new tensor factors, or graph topology; '
                        'selected here, no held-out or behavioral evidence.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
