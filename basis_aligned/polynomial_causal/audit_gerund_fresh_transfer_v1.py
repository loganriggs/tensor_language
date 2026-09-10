"""Paired descriptive intervals for the frozen fresh-transfer experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    directory = Path(__file__).resolve().parent
    source = directory / 'GERUND_FRESH_TRANSFER_V1_RESULT.json'
    result = json.loads(source.read_text())
    reports = {}
    for index, name in enumerate(('A1', 'A2', 'P', 'G', 'C', 'R')):
        panel = result['reports'][name]
        bootstrap = PairedPanelBootstrap(16, 9111401 + index)
        swap = panel['swap']
        removal = panel['zero_removal']
        base = np.asarray(removal['base_ce_change_per_row'])
        donor = np.asarray(removal['donor_ce_change_per_row'])
        reports[name] = {
            'recovery_ci95': bootstrap.mean(swap['raw_recovery_per_row']),
            'swap_signed_ce_ci95': bootstrap.mean(swap['ce_change_per_row']),
            'swap_absolute_ce_ci95': bootstrap.mean(np.abs(swap['ce_change_per_row'])),
            'zero_signed_ce_ci95': bootstrap.mean((base + donor) / 2),
            'zero_absolute_ce_ci95': bootstrap.mean((np.abs(base) + np.abs(donor)) / 2),
            'zero_base_mean_ce': float(base.mean()),
            'zero_donor_mean_ce': float(donor.mean()),
        }
    output = {
        'schema': 'gerund.fresh_transfer.audit.v1',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'predictions_unchanged': result['predictions'],
        'reports': reports,
        'scope': '4000 paired resamples of 16 authored contexts per panel. G uses one runs/run readout, not 16 agreement contrasts. Descriptive intervals do not override capability gates or change thresholds.'
    }
    with (directory / 'GERUND_FRESH_TRANSFER_AUDIT_V1_RESULT.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
