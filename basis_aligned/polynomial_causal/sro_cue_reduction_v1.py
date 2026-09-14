"""Selected development-panel reduction of measured signed causal responses.

No weights or response coefficients are fitted. S/O/SO are frozen in the board
claim before this analysis. Full model and response generation remain external.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import torch
from even_value_factorial_v1 import reconstruct


def main():
    torch.set_num_threads(2)
    p = Path(__file__).resolve().parent
    out = p / 'SRO_CUE_REDUCTION_V1_RESULT.json'
    assert not out.exists()
    source = p / 'EVEN_VALUE_FACTORIAL_NATIVE_V1_ARTIFACT.pt'
    a = torch.load(source, weights_only=True)
    rows = json.loads((p / 'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())
    coeff = a['regional_coefficients']
    cube = a['regional_cube'] - a['regional_cube'][0]
    ce = a['newline_ce_cube'] - a['newline_ce_cube'][0]
    replay = max(float((reconstruct(coeff)-a['regional_cube']).abs().max()),
                 float((reconstruct(a['newline_coefficients'])-a['newline_ce_cube']).abs().max()))
    records = []
    for label, masks in [('scalar_only', [1]), ('additive_S_O', [1, 4]), ('S_O_SO', [1, 4, 5])]:
        reduced = torch.zeros_like(coeff)
        reduced[masks] = coeff[masks]
        prediction = reconstruct(reduced)
        regional = []
        for family in range(3):
            ix = [i for i,r in enumerate(rows['regional']) if r['family'] == family]
            assert len(ix) == 24
            truth = cube[:, ix[::2], 0] - cube[:, ix[1::2], 0]
            pred = prediction[:, ix[::2], 0] - prediction[:, ix[1::2], 0]
            scale = truth[7].norm().clamp_min(1e-8)
            errors = (pred-truth).norm(dim=1) / scale
            control_error = ((prediction[:,ix,1]-cube[:,ix,1]).norm(dim=1)
                             / cube[7,ix,1].norm().clamp_min(1e-8))
            regional.append(dict(family=family, cue_scale=float(scale),
                                 cue_error_by_corner=errors.tolist(),
                                 max_cue_error=float(errors.max()),
                                 control_error_by_corner=control_error.tolist()))
        rc = torch.zeros_like(a['newline_coefficients'])
        rc[masks] = a['newline_coefficients'][masks]
        cp = reconstruct(rc)
        natural = []
        for family in range(2):
            ix = [i for i,r in enumerate(rows['natural']) if r['family'] == family]
            errors = (cp[:,ix]-ce[:,ix]).norm(dim=1) / ce[7,ix].norm().clamp_min(1e-8)
            natural.append(dict(family=family, error_by_corner=errors.tolist()))
        records.append(dict(label=label, masks=masks, regional=regional, natural=natural))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), pred_a=replay<=1e-12,
                  pred_b=all(r['max_cue_error']<=.1 for r in records[-1]['regional']),
                  replay_maxabs=replay, records=records,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Selected development-panel response analysis; no OOD or autonomous executable extraction. Three of seven measured response coefficients retained; original model and coefficient generators remain charged.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
    print(json.dumps([dict(label=r['label'],cue_errors=[f['max_cue_error'] for f in r['regional']],
                           max_control_error=[max(f['control_error_by_corner']) for f in r['regional']],
                           max_natural_error=[max(f['error_by_corner']) for f in r['natural']]) for r in records]))


if __name__ == '__main__':
    main()
