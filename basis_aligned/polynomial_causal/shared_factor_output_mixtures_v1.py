"""Exact output-mixture search at fixed shared readers and quadratic group span.

A: projection Gram, generalized eigen residual, and CP identities <=1e-9.
B: >=3/12 readers have a >=95% shared-factor mixture in the full group span.
C: >=1 qualifying mixture improves best single group by .1 and has >=2
unit-normalized group coefficients with magnitude >=.2.
Local-consumer versus full-span solves are the grouping red-team.
"""
import hashlib
import json
from pathlib import Path

import torch
from ll1_joint_parent_graph_v2 import factors
from ll1_group_matching_v1 import group_inner
from structured_branch_amplitudes_v1 import inner


def solve(gram, projected, indices):
    ids = torch.tensor(indices, dtype=torch.long)
    g = gram[ids][:, ids]
    h = projected[ids][:, ids]
    values, vectors = torch.linalg.eigh(g)
    keep = values > values.max()*1e-12
    whiten = vectors[:, keep]/values[keep].sqrt()
    small = whiten.T@h@whiten
    eigenvalues, eigenvectors = torch.linalg.eigh((small+small.T)/2)
    coeff = whiten@eigenvectors[:, -1]
    fraction = (coeff@h@coeff)/(coeff@g@coeff)
    residual = float((h@coeff-fraction*(g@coeff)).norm() /
                     (h.norm()*coeff.norm()).clamp_min(1e-30))
    full = gram.new_zeros(len(gram)); full[ids] = coeff
    return full, float(fraction), residual, int(keep.sum()), float(values.max()/values[keep].min())


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    root = Path(__file__).parent
    source = root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt'
    output = root/'SHARED_FACTOR_OUTPUT_MIXTURES_V1.json'
    artifact = root/'SHARED_FACTOR_OUTPUT_MIXTURES_V1.pt'
    assert not output.exists() and not artifact.exists()
    graph = torch.load(source, weights_only=True, map_location='cpu')
    a, s, c = factors(graph)
    scalar = a.new_ones(len(a), 1)
    gram = group_inner((a, s, scalar), (a, s, scalar))
    scales = gram.diag().sqrt()
    s = s/scales[:, None]
    gram = gram/(scales[:, None]*scales[None])
    checks, rows, saved = [], [], []
    for parent, reader in enumerate(graph['readers']):
        u = reader/reader.norm()
        qu = torch.einsum('grd,gr->gd', a, s*(a@u))
        alpha = qu@u
        projected = 2*(qu@qu.T)-alpha[:, None]*alpha[None]
        consumers = [g for g, group in enumerate(graph['groups']) if parent in group['parent_ids'].tolist()]
        # Dense matrices for two actual native groups independently verify the
        # small projected Gram, including a cross-group entry.
        dense = [(a[g].T*s[g])@a[g] for g in consumers[:2]]
        projs = [u[:, None]*(q@u)[None]+(q@u)[:, None]*u[None]-(u@q@u)*u[:, None]*u[None] for q in dense]
        for i in range(len(dense)):
            for j in range(len(dense)):
                checks.append(float(abs((projs[i]*projs[j]).sum()-projected[consumers[i], consumers[j]])))
                checks.append(float(abs((dense[i]*dense[j]).sum()-gram[consumers[i], consumers[j]])))
        solutions = {}
        for label, ids in [('consumers', consumers), ('all_groups', list(range(len(a))))]:
            coeff, fraction, residual, rank, condition = solve(gram, projected, ids)
            checks.append(residual)
            partner = 2*(coeff@qu)-(coeff@alpha)*u
            mixed = (a.flatten(0, 1), a.flatten(0, 1), (coeff[:, None]*s).flatten()[None])
            product = (u[None], partner[None], a.new_ones(1, 1))
            en = coeff@gram@coeff
            shared_energy = inner(product, product)
            cross = inner(mixed, product)
            checks.extend([float(abs(shared_energy/en-fraction)), float(abs(cross-shared_energy)/en)])
            normed = coeff/coeff.norm()
            single = float(projected.diag().max())
            solutions[label] = dict(shared_factor_fraction=fraction,
                best_single_group_fraction=single, gain_over_best_single=fraction-single,
                groups_ge_point2=int((normed.abs() >= .2).sum()),
                normalized_group_coefficients=normed.tolist(), generalized_residual=residual,
                retained_gram_rank=rank, gram_condition=condition)
            saved.append(dict(parent=parent, scope=label, coefficients=coeff,
                              reader=u, partner=partner))
        rows.append(dict(parent=parent, consumer_ids=consumers, **solutions))
    good = [r for r in rows if r['all_groups']['shared_factor_fraction'] >= .95]
    mixed = [r for r in good if r['all_groups']['groups_ge_point2'] >= 2
             and r['all_groups']['gain_over_best_single'] >= .1]
    torch.save(dict(mixtures=saved, source_group_norms=scales), artifact)
    result = dict(pred_a=max(checks)<=1e-9, pred_b=len(good)>=3, pred_c=len(mixed)>=1,
        maximum_replay=max(checks), qualifying_readers=len(good), qualifying_mixed_readers=len(mixed),
        rows=rows, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        scope='Exact fixed-reader/group-span restriction; output mixture discovery only. '
              'No topology refit, lower-cost whole-program claim, data fitting, or circuit identification.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    for r in rows:
        print(json.dumps(dict(parent=r['parent'], local=r['consumers']['shared_factor_fraction'],
            all_fraction=r['all_groups']['shared_factor_fraction'], gain=r['all_groups']['gain_over_best_single'],
            groups=r['all_groups']['groups_ge_point2'])))


if __name__ == '__main__': main()
