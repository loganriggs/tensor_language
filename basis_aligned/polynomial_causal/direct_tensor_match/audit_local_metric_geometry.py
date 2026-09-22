"""Frozen local-feature Gram geometry: no fits or candidate selection."""
import json
import time
import torch
from audit_conditional_residual_accounting import P, load
from mixed_gaussian_cp import gram_dynamic


def whitened_metric(reference, other):
    L = torch.linalg.cholesky(reference)
    left = torch.linalg.solve_triangular(L, other, upper=False)
    result = torch.linalg.solve_triangular(L, left.T, upper=False).T
    return (result + result.T) / 2


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    start = time.monotonic()
    # Known generalized eigenvalues check, including a non-diagonal reference.
    L = torch.tensor([[2., 0.], [.7, 1.]], dtype=torch.float64)
    A = L @ L.T
    B = L @ torch.diag(torch.tensor([.2, 3.], dtype=torch.float64)) @ L.T
    assert torch.allclose(torch.linalg.eigvalsh(whitened_metric(A, B)),
                          torch.tensor([.2, 3.], dtype=torch.float64), atol=1e-12)
    cache = torch.load(P / 'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt', weights_only=True)
    S = cache['projections']['covariance']['whitener'].double()
    mu = cache['mean'].double()
    cal = torch.cat([
        torch.load(P / 'NATIVE_QUARTIC_COVARIANCE_V1.pt', weights_only=True)['panels'][0]['rows'],
        torch.load(P / 'QUARTIC_ADDITIONAL_STATES_V1.pt', weights_only=True)['rows']]).double()
    fresh = torch.load(P / 'RESIDUAL_FRESH_STATES_V1.pt', weights_only=True)
    x = fresh['rows'].double()
    sensitivity = torch.load(P / 'SENSITIVE_ROOT_CALIBRATION_V2.pt', weights_only=True)['panels'][0]['weight'].double()
    sensitivity /= sensitivity.mean(0)
    assert cal.shape == (6144, 1152) and x.shape == (16384, 1152)
    rows = []
    for seed in [25001, 25002]:
        p, sha = load(f'LOCAL_QUARTIC_RESIDUAL_ADAM_SEED{seed}_V1.pt')
        for g in range(12):
            f = [a[g*8:(g+1)*8] for a in p['factors']]
            fw, b = [a @ S for a in f], [a @ mu for a in f]
            gaussian = gram_dynamic(fw, b, fw, b)
            phi = torch.stack([cal @ a.T for a in f]).prod(0)
            psi = torch.stack([x @ a.T for a in f]).prod(0)
            empirical = phi.T @ phi / len(phi)
            weighted = phi.T @ (sensitivity[:, g+4, None] * phi) / len(phi)
            evaluation = psi.T @ psi / len(psi)
            spectra = {}
            for name, ref, other in [('calibration_over_gaussian', gaussian, empirical),
                                     ('sensitive_over_gaussian', gaussian, weighted),
                                     ('evaluation_over_calibration', empirical, evaluation)]:
                eig = torch.linalg.eigvalsh(whitened_metric(ref, other))
                assert torch.isfinite(eig).all() and eig.min() > 0
                spectra[name] = eig.tolist()
            hybrid_eig = torch.linalg.eigvalsh(whitened_metric(gaussian, .5*(gaussian+weighted)))
            assert torch.allclose(hybrid_eig, .5 + .5*torch.tensor(spectra['sensitive_over_gaussian'], dtype=torch.float64), atol=1e-9, rtol=1e-9)
            rows.append(dict(seed=seed, output=g+4, sha256=sha, spectra=spectra,
                             hybrid_min_eigenvalue=float(hybrid_eig.min())))
    summary = {key:dict(min_eigenvalue=min(r['spectra'][key][0] for r in rows),
                        max_eigenvalue=max(r['spectra'][key][-1] for r in rows),
                        subspaces_within_factor_two=sum(r['spectra'][key][0] >= .5 and r['spectra'][key][-1] <= 2 for r in rows))
               for key in rows[0]['spectra']}
    result = dict(rows=rows, summary=summary, seconds=time.monotonic()-start,
                  token_content_sha256=fresh.get('token_content_sha256'),
                  controls=dict(known_generalized_spectrum=True, hybrid_floor_identity=True),
                  scope='24 frozen 8-dimensional local quartic spans. Exact noncentral Gaussian moments versus opened empirical metrics. No target residual included in these spans, no fit or export, no sample-concentration theorem or causal/OOD claim. Generalized eigenvalues describe all linear combinations of these frozen features, not the entire polynomial class.')
    (P / 'LOCAL_METRIC_GEOMETRY_V1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
