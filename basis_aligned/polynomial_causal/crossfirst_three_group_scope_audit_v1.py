"""Diagnostic scope check on frozen fresh outcomes; no fitted scale or offset."""
from pathlib import Path
import json
import torch

P = Path(__file__).resolve().parent


def main():
    artifact = torch.load(P / 'CROSSFIRST_THREE_GROUP_FRESH_V1_ARTIFACT.pt', weights_only=True)
    cells = []
    for group in range(1, 5):
        sl = slice(24 * group, 24 * (group + 1))
        z = artifact['readouts'][sl, :, 0].double()
        m = artifact['measures'][sl, :, 0].double()
        prediction = z[:, 3] - z[:, 0]
        targets = {
            'native_head17_2_conditional_write': z[:, 1] - z[:, 0],
            'full_local_block17_generation': m[:, 4] - z[:, 0],
            'original_child_remainder_nonadditivity': m[:, 2] - m[:, 1] - m[:, 3] + m[:, 0],
        }
        metrics = {}
        for name, target in targets.items():
            assert target.norm() > 0
            metrics[name] = {
                'relative_error': float((prediction - target).norm() / target.norm()),
                'maximum_absolute_margin_error': float((prediction - target).abs().max()),
                'zero_predictor_relative_error': 1.0,
            }
        cells.append({'group': group, 'targets': metrics})
    result = {
        'cells': cells,
        'scope': 'Diagnostic, no fitted scaling: different conditional backgrounds prevent interpreting these errors as unique causal coverage. Small local prediction error is not total-interaction sufficiency.',
    }
    (P / 'CROSSFIRST_THREE_GROUP_SCOPE_V1_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
