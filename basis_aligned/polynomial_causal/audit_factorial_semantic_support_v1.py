"""Replay opened screens and distinguish cross-token from lexical interactions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import factorial_semantic_support_v1 as F

P = Path(__file__).resolve().parent


def main():
    inputs = {}
    def read(name):
        path = P / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    oldrows = read('SUBJECT_OBJECT_CONTROLLER_V1_ROWS.json')
    old = read('SUBJECT_OBJECT_CONTROLLER_V1_RESULT.json')
    rows = read('THIRD_NOUN_ANIMACY_V1_ROWS.json')
    result = read('THIRD_NOUN_ANIMACY_V1_RESULT.json')
    assert old['predictions']['pred_a_instrument'] and result['predictions']['pred_a_instrument']
    errors = []
    oldcoef = np.array([F.signed_coefficients(oldrows['corners'], m) for m in old['margins']])
    # Parent uses an explicit non-mask term order. Map names, not positions.
    term_masks = [0 if t == '1' else sum(1 << 'cso'.index(x) for x in t) for t in oldrows['terms']]
    for split, report in old['reports'].items():
        indices = [i for i, w in enumerate(oldrows['worlds']) if w['split'] == split]
        errors.append(float(np.max(abs(oldcoef[indices][:, term_masks] - report['margin_coefficients']))))
    coef = np.array([F.signed_coefficients(rows['corners'], m) for m in result['margins']])
    errors.append(float(np.max(abs(coef - result['margin_coefficients']))))
    assert max(errors) < 1e-12
    # Row order is not part of the transform contract.
    perm = np.random.default_rng(19).permutation(32)
    perm_error = float(np.max(abs(F.signed_coefficients(
        [rows['corners'][i] for i in perm], np.array(result['margins'])[0, perm]) - coef[0])))
    assert perm_error == 0
    rules = {}
    for rule, report in result['reports'].items():
        labels = [r['labels'][rule] for r in rows['worlds'][0]['rows']]
        replay = F.score_rule(result['margins'], labels)
        assert replay['passed'] == report['passed'] and replay['raw_correct'] == report['raw_correct']
        assert np.max(abs(np.array(replay['signed_mean_margin']) -
                          [c['signed_mean_margin'] for c in report['cells']])) < 1e-12
        assert replay['accuracy'] == [c['accuracy'] for c in report['cells']]
        rules[rule] = {'passed': replay['passed'], 'raw_correct': replay['raw_correct']}
    support = {}
    for pair, i, j in [('oh', 2, 4), ('ah', 3, 4)]:
        squares = [s for w in rows['worlds'] for s in F.token_square_support(w['rows'], i, j)]
        support[pair] = {'squares': len(squares),
            'universal_token_local_zero_count': sum(s['universal_token_local_zero'] for s in squares),
            'first_edit_position_sets': sorted({tuple(s['first_edit_positions']) for s in squares}),
            'second_edit_position_sets': sorted({tuple(s['second_edit_positions']) for s in squares}),
            'unmatched_position_sets': sorted({tuple(s['unmatched_diagonal_positions']) for s in squares})}
    assert support['oh']['universal_token_local_zero_count'] == 64
    assert support['ah']['universal_token_local_zero_count'] == 0
    stats = {t: {'mean': float(coef[:, i].mean()), 'rms': float(np.sqrt((coef[:, i]**2).mean()))}
             for i, t in enumerate(rows['terms'])}
    # A mixed output does not imply multiplicative message formation:
    # additive residual changes alone suffice under a normalized decoder.
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    states = np.array([[1 + o/4, 1 + h/3] for o, h in corners])
    raw = F.signed_coefficients(corners, states[:, 0])
    normalized = F.signed_coefficients(corners, states[:, 0]/np.sqrt((states**2).mean(1)))
    assert raw[3] == 0 and abs(normalized[3]) > .001
    receipt = {'passed': True, 'source_hashes': inputs, 'model_forwards': 0,
        'coefficient_replay_max_abs': max(errors), 'permuted_row_error': perm_error,
        'rule_replay': rules, 'support_certificates': support, 'coefficient_statistics': stats,
        'oh_to_o_rms_ratio': stats['oh']['rms']/stats['o']['rms'],
        'normalized_decoder_counterexample': {'raw_mixed': float(raw[3]),
                                              'normalized_mixed': float(normalized[3])},
        'shared_helper_sha256': hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),
        'walsh_kernel_sha256': hashlib.sha256((P/'dealiased_boolean_spectrum.py').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Post-outcome finite-domain diagnostic, not an identified circuit or held-out predictor. '
                 'oh must arise after token-local lookup, but may be decoder-created. ah can originate in lookup.'}
    with (P/'FACTORIAL_SEMANTIC_SUPPORT_V1_RESULT.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('source_hashes', 'coefficient_statistics')}, indent=2))


if __name__ == '__main__':
    main()
