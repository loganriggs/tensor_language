"""Sharp four-corner bound for additive output models, not internal programs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def best_additive(cube):
    interaction = cube[0, 0]-cube[1, 0]-cube[0, 1]+cube[1, 1]
    signs = np.array([[1., -1.], [-1., 1.]])
    signs = signs.reshape((2, 2)+(1,)*(cube.ndim-2))
    return cube-signs*interaction/4


def main(source, target):
    rng = np.random.default_rng(9100858)
    worst_closure = 0.; worst_attainment = 0.
    for _ in range(32):
        cube = rng.normal(size=(2, 2, 7, 9))
        interaction = cube[0, 0]-cube[1, 0]-cube[0, 1]+cube[1, 1]
        fitted = best_additive(cube)
        worst_closure = max(worst_closure, float(np.max(np.abs(fitted[0, 0]-fitted[1, 0]-fitted[0, 1]+fitted[1, 1]))))
        bound = np.linalg.norm(interaction)/4
        errors = np.linalg.norm((cube-fitted).reshape(4, -1), axis=1)
        worst_attainment = max(worst_attainment, float(np.max(np.abs(errors-bound))))
    assert max(worst_closure, worst_attainment) < 1e-12
    parent = json.loads(source.read_text()); assert parent['predictions']['pred_a_instrument']
    reports = []
    for row in parent['reports']:
        # Linear centering and Q commute with the four-corner difference.
        # The parent's stored norm ratio therefore suffices; no missing full
        # vocabulary vectors are reconstructed or claimed to have been saved.
        bounds = {scope: {metric: value/4 if value is not None else None
                         for metric, value in ratios.items()}
                  for scope, ratios in row['interaction_errors'].items()}
        reports.append({'world_id': row['world_id'], 'layout': row['layout'],
                        'sharp_worst_corner_bound_over_coupling_total': bounds})
    values = [r['sharp_worst_corner_bound_over_coupling_total']['mixed']['vocabulary'] for r in reports]
    result = {'parent_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'controls': {'passed': True, 'random_fixtures': 32,
                           'additive_closure_max_abs': worst_closure,
                           'bound_attainment_max_abs': worst_attainment},
              'reports': reports,
              'mixed_vocabulary_bound_range': [min(values), max(values)],
              'worlds_excluding_uniform_10_percent_additive_output_error': sum(v > .10 for v in values),
              'scope': 'Sharp minimax norm bound for any Y_ab=B+F(a)+G(b), normalized by the parent directional-coupling total, on these fixed intervention tables. Does not rule out shared nonlinear readers, internal additive writes, context-dependent joint operations, or smaller nonlinear programs. Native measurements are numerical, not exact arithmetic certificates of the trained model.'}
    with target.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'reports'}))


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]), Path(sys.argv[2]))
