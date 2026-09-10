"""Opened-output nuisance attribution and exact contextual attention identity."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from mixed_state_projector_v1 import projector


def main():
    directory = Path(__file__).resolve().parent
    source = directory / 'THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json'
    data = json.loads(source.read_text())
    assert data['predictions']['pred_a_instrument']
    q = projector(data['corners'], 2, 4)
    complement = np.eye(32) - q
    rows = []
    for world in data['reports']:
        native = np.array(world['native_logits'])
        foil = 1 if world['foil'] == 'himself' else 2
        effects = {}
        for name, logits in world['arm_logits'].items():
            dz = native - np.array(logits)
            effects[name] = dz[:, 0] - dz[:, foil]
        effects['interaction'] = effects['full'] - effects['mixed'] - effects['spill']
        nuisance = {name: complement @ effect for name, effect in effects.items()}
        full = nuisance['full']
        projections = {name: float(v @ full / (full @ full))
                       for name, v in nuisance.items() if name != 'full'}
        assert abs(sum(projections.values()) - 1) < 1e-12
        rows.append({'world_id': world['world_id'],
                     'nuisance_signed_projections': projections,
                     'mixed_only_remaining_nuisance_norm_ratio': float(np.linalg.norm(nuisance['mixed']) / np.linalg.norm(full)),
                     'mixed_natural_projection': world['metrics']['mixed']['live_natural_mixed_projection'],
                     'mixed_gender_ratio': world['mixed_gender_ratio'],
                     'composition_reader_error': world['composition_reader_error']})

    # A conditional four-corner matrix-valued routing operator: no commutation
    # between unrelated matrices is assumed. The row projector acts on factors.
    rng = np.random.default_rng(910643)
    corners = list(itertools.product((-1, 1), repeat=2))
    routing = rng.normal(size=(4, 3, 5))
    value_coefficient = rng.normal(size=5)
    d = np.array([o*h*value_coefficient for o,h in corners])
    output = np.einsum('nij,nj->ni', routing, d)
    q2 = projector(corners, 0, 1)
    folded = np.einsum('ij,nj->ni', routing.mean(axis=0), d)
    error = float(np.max(abs(q2 @ output - folded)))
    assert error < 1e-12
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'routing_identity': {'formula': 'Q_oh(P d)=mean_oh(P) d when d(o,h)=oh d_oh',
                                   'max_absolute_error': error, 'passed': True},
              'reports': rows,
              'scope': 'Signed nuisance contributions on opened outcomes; cancellation is allowed. Synthetic matrix identity validates algebra, not independent native production or gender selectivity.'}
    with (directory / 'THIRD_NOUN_WRITE_FACTOR_ATTRIBUTION_V1_RESULT.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
