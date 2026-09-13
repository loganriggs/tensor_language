"""Unfiltered per-prefix effect diagnostics for existing and folded interactions."""
import argparse
import json
from pathlib import Path
import torch

P = Path(__file__).resolve().parent


def compare(predicted, reference):
    difference = predicted-reference
    nz = reference != 0
    opposite = predicted*reference < 0
    norm = reference.norm()
    return dict(relative_l2=float(difference.norm()/norm) if norm > 0 else None,
                reference_norm=float(norm), maxabs_error=float(difference.abs().max()),
                meanabs_error=float(difference.abs().mean()),
                same_nonzero_sign=int((predicted*reference > 0).sum()),
                opposite_sign=int(opposite.sum()),
                reference_nonzero_predicted_zero=int((nz & (predicted == 0)).sum()),
                reference_zero_predicted_nonzero=int(((~nz) & (predicted != 0)).sum()),
                both_zero=int(((~nz) & (predicted == 0)).sum()),
                maxabs_opposite_reference=float(reference[opposite].abs().max()) if opposite.any() else 0.,
                worst_absolute_row=int(difference.abs().argmax()),
                reference_effects=reference.tolist(), predicted_effects=predicted.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stem', choices=['MLP10_MIXED_INPUT_SOURCES_V1',
                                        'MLP9_TO_MLP10_RESIDUAL_FOLD_V1'])
    args = parser.parse_args()
    measures = torch.load(P/(args.stem+'_ARTIFACT.pt'), weights_only=True,
                          map_location='cpu')['measures'].double()
    assert measures.shape[0] == 160 and measures.shape[2] == 2
    pairs = [('residual_only_to_cross', 11, 8), ('mixed_only_to_cross', 12, 8)]
    if args.stem.startswith('MLP9_'):
        assert measures.shape[1] == 17
        pairs += [('joint_to_cross', 14, 8), ('folded_to_joint', 15, 14),
                  ('folded_joint_to_cross', 15, 8), ('folded_full_to_cross', 16, 8)]
    cells = []
    groups = [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]
    groups += [(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]
    for lo, hi, label in groups:
        for endpoint in range(2):
            z = measures[lo:hi, :, endpoint]
            for name, prediction, reference in pairs:
                cells.append(dict(cell=label, endpoint=endpoint, comparison=name,
                                  first_global_row=lo,
                                  **compare(z[:, prediction]-z[:, 5],
                                            z[:, reference]-z[:, 5])))
    result = dict(cells=cells, scope='All 160 prefixes, both endpoints, no magnitude filtering. '
                  'Regional endpoint0 UK-US margin; endpoint1 control-token margin. '
                  'FineWeb endpoint0 newline CE; endpoint1 newline-comma margin. '
                  'Sign discrepancies and tiny absolute errors are reported together. '
                  'No precision-floor attribution or promotion beyond registered criteria.')
    target = P/(args.stem+'_OUTCOME_PAIRS_AUDIT.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in
                      ('reference_effects','predicted_effects')}
                     for c in cells if c['endpoint'] == 0], indent=2))


if __name__ == '__main__':
    main()
