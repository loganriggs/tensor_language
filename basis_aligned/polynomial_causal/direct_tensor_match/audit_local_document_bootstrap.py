"""Descriptive paired whole-document bootstrap; no fitting or promotion gate."""
import json
import time
import torch
from audit_conditional_residual_accounting import P, SCALE, load
from audit_local_quartic_followup import correction


def statistics(sse, energy):
    relative = (sse / energy).sqrt()
    return relative, relative[..., 4:].square().mean(-1).sqrt()


def interval(values):
    return torch.quantile(values, torch.tensor([.025, .975], dtype=values.dtype), dim=0).tolist()


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    data = torch.load(P / 'RESIDUAL_FRESH_STATES_V1.pt', weights_only=True)
    x, y = data['rows'].double(), data['target'].double() / SCALE
    assert x.shape == (16384, 1152) and y.shape == (16384, 16)
    parent, parent_sha = load('MIXED_CP_FEATURES_SEED1001_V1.pt')
    base = torch.cat([
        torch.stack([xx @ f.T for f in parent['factors']]).prod(0)
        @ parent['coefficients'].T / SCALE for xx in x.split(1024)])
    energy = y.reshape(256, 64, 16).square().sum(1)
    base_sse = (base - y).reshape(256, 64, 16).square().sum(1)
    generator = torch.Generator().manual_seed(39221)
    draws = torch.randint(256, (2000, 256), generator=generator)
    counts = torch.zeros(2000, 256, dtype=torch.float64)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    assert torch.equal(counts.sum(1), torch.full((2000,), 256., dtype=torch.float64))
    # Count representation must equal explicitly repeated sampled documents.
    assert torch.allclose(counts[0] @ base_sse, base_sse[draws[0]].sum(0), rtol=1e-12)
    boot_energy = counts @ energy
    _, base_boot = statistics(counts @ base_sse, boot_energy)
    base_per, base_point = statistics(base_sse.sum(0), energy.sum(0))
    followup = json.loads((P / 'LOCAL_QUARTIC_FOLLOWUP_V1.json').read_text())
    rows = []
    for seed in [25001, 25002]:
        candidate, sha = load(f'LOCAL_QUARTIC_RESIDUAL_ADAM_SEED{seed}_V1.pt')
        assert candidate['parent_sha256'] == parent_sha
        pred = base.clone()
        pred[:, 4:] += torch.cat([correction(candidate, xx) for xx in x.split(1024)])
        sse = (pred - y).reshape(256, 64, 16).square().sum(1)
        per, point = statistics(sse.sum(0), energy.sum(0))
        boot_per, boot = statistics(counts @ sse, boot_energy)
        ref = next(r for r in followup['rows'] if r['optimizer'] == 'adam' and r['seed'] == seed)
        assert sha == ref['sha256'] and abs(float(point) - ref['small_value_rms']) < 1e-10
        assert torch.equal(sse[:, :4], base_sse[:, :4])
        row = dict(seed=seed, sha256=sha, small_value_rms=float(point),
                   small_value_rms_interval=interval(boot),
                   relative_improvement=float(1 - point / base_point),
                   relative_improvement_interval=interval(1 - boot / base_boot),
                   per_output_errors=per.tolist(), per_output_intervals=interval(boot_per),
                   per_output_relative_improvement=(1 - per / base_per).tolist(),
                   per_output_relative_improvement_intervals=interval(
                       1 - boot_per / statistics(counts @ base_sse, boot_energy)[0]))
        rows.append(row)
        print(seed, row['relative_improvement'], row['relative_improvement_interval'], flush=True)
    result = dict(rows=rows, baseline_small_value_rms=float(base_point),
                  baseline_interval=interval(base_boot), parent_sha256=parent_sha,
                  token_content_sha256=data.get('token_content_sha256'),
                  bootstrap_seed=39221, replicates=2000, documents=256, positions_per_document=64,
                  controls=dict(counts_equal_explicit_resampling=True, score_replay=True,
                                protected_outputs_exact=True),
                  seconds=time.monotonic() - start,
                  scope='Paired percentile 95% document-bootstrap intervals, conditional on frozen fits and opened panel. Documents are assumed exchangeable sampling units. No refitting, training-seed uncertainty, OOD claim, selection correction, simultaneous per-output coverage, response-pair interval, or causal adoption. Same document resamples for parent and both candidates.')
    (P / 'LOCAL_DOCUMENT_BOOTSTRAP_V1.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
