"""Expose response rotations hidden by aggregate effect-norm matching."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/expanded_controls_equal_strength_v1_result.json'
a=json.loads(source.read_text());records=[]
for r in a['records']:
    c=np.asarray(r['candidate_effect'])[:,0];b=np.asarray(r['baseline_effect'])[:,0]
    coefficient=float(c@b/(b@b));parallel=coefficient*b;orthogonal=c-parallel
    records.append(dict(panel=r['panel'],family=r['family'],role=r['role'],norm_relative_error=r['strength_error'],cosine=r['number_cosine'],parallel_scale=coefficient,orthogonal_fraction=float(np.linalg.norm(orthogonal)/np.linalg.norm(c)),sign_disagreements=int(np.sum(c*b<0)),orthogonality_error=float(abs(orthogonal@b))))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),max_norm_relative_error=max(r['norm_relative_error'] for r in records),records=records,scope='Projection of measured response vectors, not a model feature decomposition. A matching norm alone cannot justify collateral improvement for different target responses.')
(P/'NUMBER_EFFECT_ROTATION_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(max_norm_error=out['max_norm_relative_error'],worst_orthogonal=max(r['orthogonal_fraction'] for r in records),sign_disagreements=sum(r['sign_disagreements'] for r in records),orthogonality_error=max(r['orthogonality_error'] for r in records))))
