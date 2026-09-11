"""Weight-defined spherical-constant control; no data-fitted coefficients.

A dense and split replay <=1e-9. B radial relative RMS error <=.25 for each
branch on both frozen panels. C sign agreement >=.9 in each case.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    output = root / 'BRANCH_RADIAL_CONTROL_V1.json'
    assert not output.exists()
    source = root / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    node = torch.load(source, weights_only=True, map_location='cpu')['nodes'][1]
    u, p = node['reader'].double(), node['partners'].double()
    d = u.numel()
    alpha = u @ p / d
    matrices = torch.stack([(torch.outer(u, p[:, j]) + torch.outer(p[:, j], u))/2 for j in range(2)])
    traceless = matrices - alpha[:, None, None] * torch.eye(d)
    coefficient_fraction = d * alpha.square() / matrices.square().sum((1, 2))
    records, sources, errors = {}, [source], []
    for label, stem in [('fineweb', 'SUPPRESSION_V1'), ('corpus_shift', 'CORPUS_SHIFT_V1')]:
        path = root / f'SHARED_NODE_PARENT1_{stem}_ENDPOINTS.pt'
        panel_path = root / f'SHARED_NODE_PARENT1_{stem}_ROWS.pt'
        sources += [path, panel_path]
        cache = torch.load(path, weights_only=True, map_location='cpu')
        panel = torch.load(panel_path, weights_only=True, map_location='cpu')
        x = cache['ports']['input'].double()
        n = len(panel['documents'])
        assert x.shape == (3*n, d)
        actual = (x @ u)[:, None] * (x @ p)
        radial = x.square().sum(1)[:, None] * alpha
        dense = torch.einsum('nd,kde,ne->nk', x, matrices, x)
        nonradial = torch.einsum('nd,kde,ne->nk', x, traceless, x)
        errors += [float((actual-dense).norm()/actual.norm()),
                   float((actual-radial-nonradial).norm()/actual.norm())]

        def metrics(ids):
            a, r = actual[ids], radial[ids]
            norm = a.square().mean(0).sqrt()
            assert bool((norm > 0).all()) and bool(torch.isfinite(a).all())
            return dict(relative_rms_error=((a-r).square().mean(0).sqrt()/norm).tolist(),
                radial_to_actual_rms=(r.square().mean(0).sqrt()/norm).tolist(),
                sign_agreement=(a.sign()==r.sign()).double().mean(0).tolist(),
                actual_mean=a.mean(0).tolist(), actual_rms=norm.tolist(),
                radial_mean=r.mean(0).tolist(), endpoints=len(a))

        records[label] = dict(overall=metrics(torch.arange(3*n)),
            families=[metrics(torch.arange(i*n, (i+1)*n)) for i in range(3)])
        if 'domain_labels' in panel:
            records[label]['domains'] = {domain: metrics(torch.tensor([
                i for i in range(3*n) if panel['domain_labels'][i % n] == domain]))
                for domain in sorted(set(panel['domain_labels']))}
    a = max(errors) <= 1e-9
    result = dict(pred_a=a,
        pred_b=a and all(max(r['overall']['relative_rms_error']) <= .25 for r in records.values()),
        pred_c=a and all(min(r['overall']['sign_agreement']) >= .9 for r in records.values()),
        replay_errors=errors, alpha=alpha.tolist(), coefficient_radial_energy_fraction=coefficient_fraction.tolist(),
        panels=records,
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scope='Post-result diagnostic on frozen validation caches; exact spherical-constant term from weights. '
              'No fitted mean, no behavioral tail replay, no claim that remaining variation is meaningful context.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','panels')}, indent=2))
    print(json.dumps({k:v['overall'] for k,v in records.items()}, indent=2))


if __name__ == '__main__':
    main()
