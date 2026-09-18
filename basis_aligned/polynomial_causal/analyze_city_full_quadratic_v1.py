"""Opened local quadratic-write diagnostic; cannot establish suffix necessity."""
import json
from pathlib import Path
import torch

P = Path(__file__).resolve().parent
torch.set_num_threads(2)
half = torch.load(P/'CITY_FULL_PILE_V3_ARTIFACT.pt', weights_only=True)['fixtures']
full = torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt', weights_only=True)['fixtures']
rows = []
for i, (a, b) in enumerate(zip(half, full, strict=True)):
    assert torch.equal(a['candidate_inputs']['residual7'], b['candidate_inputs']['residual7'])
    assert torch.equal(a['candidate_inputs']['token_ids'], b['candidate_inputs']['token_ids'])
    h, f = a['expected_candidate_delta'], b['expected_candidate_delta']
    q = f - 2*h
    rows.append({'sequence': i, 'quadratic_correction_over_full_norm': float(q.norm()/f.norm()),
                 'aligned_fraction': float((q*f).sum()/f.square().sum())})
out = {'scope': 'Opened local write diagnostic; full minus twice half isolates quadratic strength dependence; no suffix attribution',
       'sequences': rows,
       'max_norm_ratio': max(r['quadratic_correction_over_full_norm'] for r in rows),
       'min_norm_ratio': min(r['quadratic_correction_over_full_norm'] for r in rows)}
(P/'CITY_FULL_QUADRATIC_V1_CPU_RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='sequences'}))
