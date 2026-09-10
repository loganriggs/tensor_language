"""Opened-result diagnostic: number/gender geometry, with no new fitted operator."""
import hashlib
import json
from pathlib import Path

import numpy as np
from mixed_state_projector_v1 import projector


def main():
    directory = Path(__file__).resolve().parent
    source = directory / 'THIRD_NOUN_VALUE_TRANSFER_V1_RESULT.json'
    data = json.loads(source.read_text())
    assert data['predictions']['pred_a_instrument']
    q = projector(data['corners'], 2, 4)
    reports = []
    for world in data['reports']:
        delta = np.array(world['native_logits']) - np.array(world['edited_logits'])
        mixed = q @ delta
        number = mixed[:, 0] - (mixed[:, 1] + mixed[:, 2]) / 2
        gender = mixed[:, 1] - mixed[:, 2]
        male = mixed[:, 0] - mixed[:, 1]
        female = mixed[:, 0] - mixed[:, 2]
        centered = mixed - mixed.mean(axis=1, keepdims=True)
        number_energy = (2 / 3) * float(number @ number)
        gender_energy = .5 * float(gender @ gender)
        energy_error = abs(float(np.sum(centered**2)) - number_energy - gender_energy)
        pair_error = max(float(np.max(abs(male - (number - gender / 2)))),
                         float(np.max(abs(female - (number + gender / 2)))))
        assert energy_error < 1e-12 and pair_error < 1e-12
        metrics = world['metrics']
        spill = np.array(metrics['spill_masks'])
        coefficients = np.array(metrics['effect_coefficients'])
        largest = int(spill[np.argmax(abs(coefficients[spill]))])
        factors = ('c', 's', 'o', 'a', 'h')
        reports.append({
            'world_id': world['world_id'],
            'number_fraction_centered_mixed_reader_energy': number_energy / (number_energy + gender_energy),
            'male_female_effect_cosine': float(male @ female / (np.linalg.norm(male) * np.linalg.norm(female))),
            'energy_identity_absolute_error': energy_error,
            'answer_pair_identity_absolute_error': pair_error,
            'largest_spill_factor': ''.join(f for i, f in enumerate(factors) if largest & (1 << i)) or 'constant',
            'failed_bars': (["native_floor"] if world['native_mixed_margin_rms'] < .05 else [])
                + (["materiality"] if metrics['live_natural_mixed_projection'] < .05 else [])
                + (["factor_spill"] if metrics['nonmixed_to_mixed_margin_ratio'] > .25 else [])
                + (["gender_control"] if not world['gender_selectivity_passed'] else []),
        })
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reports': reports,
              'scope': 'Descriptive geometry of opened output effects. Energy concentration does not override failed causal selectivity or identify shared hidden variables. No fit, new forward, or changed threshold.'}
    target = directory / 'THIRD_NOUN_READER_GEOMETRY_V1_RESULT.json'
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
