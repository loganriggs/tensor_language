"""Opened descriptive diagnostic; no new gates, fitting, or row selection."""
import hashlib
import json
from pathlib import Path

import torch

P = Path(__file__).resolve().parent
torch.set_num_threads(2)
rows = json.loads((P / 'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows']
artifact = P / 'CITY_FULL_PILE_V3_ARTIFACT.pt'
v = torch.load(artifact, weights_only=True)['values']
effects = v - v[:1]
records = []
for cell in sorted({r['context_id'] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r['context_id'] == cell]
    base = v[0, idx, 0][::2] - v[0, idx, 0][1::2]
    cap = base >= .1
    atten = []
    for arm in (1, 3):
        pair = v[arm, idx, 0][::2] - v[arm, idx, 0][1::2]
        atten.append((base - pair) / base)
    native, candidate = atten
    records.append({
        'context_id': cell,
        'capable_endpoints': int(cap.sum()),
        'native_reversed_endpoints': torch.where(cap & (native <= 0))[0].tolist(),
        'candidate_reversed_endpoints': torch.where(cap & (candidate <= 0))[0].tolist(),
        'sign_disagreements': torch.where(cap & ((native > 0) != (candidate > 0)))[0].tolist(),
        'native_attenuations': native[cap].tolist(),
        'candidate_attenuations': candidate[cap].tolist(),
    })
strata = {}
for natural in (True, False):
    idx = [i for i, r in enumerate(rows) if r['is_untouched_natural_arm'] == natural]
    target = effects[1, idx, 0]
    prediction = effects[3, idx, 0]
    strata['untouched' if natural else 'substituted'] = {
        'rows': len(idx),
        'relative_effect_error': float((prediction - target).norm() / target.norm()),
        'native_target_rms': float(target.square().mean().sqrt()),
    }
out = {'scope': 'Opened descriptive diagnostic, no prospective pass/fail claim',
       'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
       'strata': strata, 'documents': records,
       'sign_disagreement_count': sum(len(r['sign_disagreements']) for r in records)}
(P / 'CITY_FULL_PILE_V3_DOCUMENT_DIAGNOSTIC.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'strata': strata, 'sign_disagreement_count': out['sign_disagreement_count'],
                  'reversal_documents': [r for r in records if r['candidate_reversed_endpoints']]}, indent=2))
