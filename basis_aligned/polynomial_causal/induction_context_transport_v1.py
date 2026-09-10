"""Read-only FIT diagnostic of R594 context transport and exact CE decomposition."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def lse(x):
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def loss_terms(z, delta, answer):
    z = np.asarray(z, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    logp = z - lse(z)
    logq = z + delta - lse(z + delta)
    p = np.exp(logp)
    mu = float(p @ delta)
    linear = mu - float(delta[answer])
    kl = float(p @ (logp - logq))
    kl_formula = lse(z + delta) - lse(z) - mu
    ce = float(logp[answer] - logq[answer])
    return dict(ce_damage=ce, linear_loss_change=linear, kl=kl,
                identity_error=abs(ce - linear - kl),
                independent_kl_error=abs(kl - kl_formula))


def controls():
    z = np.array([-3., 0., 2., 1.])
    errors = []
    for d in (np.zeros(4), np.full(4, 7.), np.array([1., -2., .5, 3.]),
              np.array([-1., 2., -.5, -3.])):
        a = loss_terms(z, d, 2)
        b = loss_terms(z, d + 12., 2)
        assert a['kl'] >= -1e-12
        assert abs(a['ce_damage'] - b['ce_damage']) < 1e-12
        assert abs(a['linear_loss_change'] - b['linear_loss_change']) < 1e-12
        errors += [a['identity_error'], a['independent_kl_error']]
    assert max(errors) < 1e-12
    return dict(cases=4, maximum_identity_error=max(errors), all_passed=True)


def main():
    poly = Path(__file__).parent
    result_path = poly / 'INDUCTION_CONTEXT_TRANSPORT_V1_RESULT.json'
    assert not result_path.exists()
    control_result = controls()
    envelope = json.loads((poly / 'INDUCTION_R594_MANAGED_RESULT.json').read_text())
    receipt = json.loads((poly / 'induction_centered_fixed_geometry_rung594_receipt.json').read_text())
    root = Path(envelope['raw_root']) / 'induction_centered_fixed_geometry_rung594_evidence'
    names = ('endpoint_records.jsonl', 'directed_records.jsonl', 'endpoint_logits.npy',
             'directed_replay_logits.npy', 'logit_differences.npy')
    bindings = {}
    for name in names:
        path = root / 'FIT' / name
        expected = receipt['evidence_files']['FIT/' + name]
        assert path.stat().st_size == expected['byte_length']
        bindings[name] = digest(path)
        assert bindings[name] == expected['sha256'], name
    phase = root / 'FIT'
    rows = [json.loads(x) for x in (phase / 'directed_records.jsonl').read_text().splitlines()]
    endpoints = [json.loads(x) for x in (phase / 'endpoint_records.jsonl').read_text().splitlines()]
    locations = {x['endpoint_id']: x['array_index'] for x in endpoints}
    native = np.load(phase / 'endpoint_logits.npy', mmap_mode='r')
    replay = np.load(phase / 'directed_replay_logits.npy', mmap_mode='r')
    diffs = np.load(phase / 'logit_differences.npy', mmap_mode='r')
    assert len(rows) == len(diffs) == len(replay) == 3744
    assert native.shape == (1728, 50304) and diffs.shape == (3744, 4, 50304)
    grouped = defaultdict(list)
    scalar_rows = []
    maximum_bridge = 0.
    for i, row in enumerate(rows):
        if not (row['family'] == 'selector_payload_joint_answer_preserved' or
                (row['family'] == 'copy_relation_preserved_nuisance_change' and row['control_kind'] == 'filler')):
            continue
        assert not row['answer_changes']
        z = np.array(replay[i], dtype=np.float64)
        delta = np.array(diffs[i, 3], dtype=np.float64)
        recipient = np.array(native[locations[row['recipient_endpoint_id']]], dtype=np.float64)
        donor = np.array(native[locations[row['donor_endpoint_id']]], dtype=np.float64)
        n = donor - recipient
        dc, nc = delta - delta.mean(), n - n.mean()
        nn, dn = float(np.linalg.norm(nc)), float(np.linalg.norm(dc))
        scale = nn / np.sqrt(len(n))
        a = loss_terms(z, delta, row['recipient_answer_id'])
        registered = row['arms']['joint']
        bridge = max(abs(a['ce_damage'] - (registered['correct_ce'] - row['replay']['correct_ce'])),
                     abs(z[row['recipient_answer_id']] + delta[row['recipient_answer_id']] - registered['answer_logit']),
                     float(np.max(np.abs(z - recipient))))
        maximum_bridge = max(maximum_bridge, bridge)
        assert bridge <= 5e-5, (row['directed_id'], bridge)
        assert a['identity_error'] < 1e-10 and a['independent_kl_error'] < 1e-10
        assert a['kl'] >= -1e-10
        item = dict(directed_id=row['directed_id'], group_id=row['group_id'],
                    native_context_rms=scale, adequate_scale=scale > 1e-4,
                    relative_context_error=float(np.linalg.norm(dc - nc) / nn) if nn else None,
                    signed_context_cosine=float(dc @ nc / (dn * nn)) if dn * nn else None,
                    context_norm_ratio=dn / nn if nn else None,
                    common_offset_energy_fraction=float(delta.mean() ** 2 / np.mean(delta ** 2)) if dn else None,
                    **a)
        key = '|'.join(str(row[k]) for k in ('family', 'variant', 'recipient_condition', 'direction'))
        grouped[key].append(item)
        scalar_rows.append(dict(cell=key, **item))
    summaries = {}
    for key, items in sorted(grouped.items()):
        assert len(items) == len({x['group_id'] for x in items}) == 72
        adequate = all(x['adequate_scale'] for x in items)
        summary = dict(groups=len(items), all_scales_adequate=adequate)
        for metric in ('relative_context_error', 'signed_context_cosine', 'context_norm_ratio',
                       'common_offset_energy_fraction', 'native_context_rms'):
            values = [x[metric] for x in items if x[metric] is not None]
            summary['median_' + metric] = float(np.median(values)) if values else None
        for metric in ('ce_damage', 'linear_loss_change', 'kl'):
            summary['mean_' + metric] = float(np.mean([x[metric] for x in items]))
        summary['context_transport_passed'] = adequate and summary['median_relative_context_error'] <= .25
        summaries[key] = summary
    assert len(scalar_rows) == 864 and len(summaries) == 12
    output_rows = poly / 'INDUCTION_CONTEXT_TRANSPORT_V1_ROWS.jsonl'
    with output_rows.open('x') as f:
        for row in scalar_rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    result = dict(experiment='induction_context_transport_v1', evaluated_split='FIT',
                  post_result_exploratory=True, original_gates_changed=False, model_forwards=0,
                  controls=control_result, raw_hashes=bindings, rows=len(scalar_rows), cells=summaries,
                  maximum_scalar_replay_bridge_error=maximum_bridge,
                  context_transport_held=all(x['context_transport_passed'] for x in summaries.values()),
                  source_sha256=digest(__file__), plan_sha256=digest(poly / 'INDUCTION_CONTEXT_TRANSPORT_V1_PLAN.md'),
                  row_sha256=digest(output_rows),
                  native_circuit_identified=False, raw_storage_volatile=True)
    with result_path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
