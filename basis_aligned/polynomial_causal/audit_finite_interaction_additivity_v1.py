"""Redteam summing independently measured finite removals as circuit terms."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/finite_interaction_census_v1_result.json'
a=json.loads(source.read_text());records=[]
for g in a['groups']:
    keys=sorted({(r['family'],tuple(r['coordinate'])) for r in g['records']})
    for family,coordinate in keys:
        rr=[r for r in g['records'] if r['family']==family and tuple(r['coordinate'])==coordinate]
        assert len(rr)==14
        target=np.asarray(rr[0]['original_interaction']);effects=np.stack([np.asarray(r['effect']) for r in rr]);prediction=effects.sum(0)
        norms=np.linalg.norm(target,axis=0);errors=np.linalg.norm(prediction-target,axis=0)
        records.append(dict(panel=g['panel'],template=g['template'],family=family,coordinate=coordinate,target_norms=norms.tolist(),sum_removal_errors=errors.tolist(),relative_errors=(errors/np.maximum(norms,1e-12)).tolist()))
summary=dict(number_pass10=sum(r['sum_removal_errors'][0]<=max(1e-8,.1*r['target_norms'][0]) for r in records),count=len(records),number_median=float(np.median([r['relative_errors'][0] for r in records])),number_worst=max(r['relative_errors'][0] for r in records))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=summary,records=records,scope='Opened diagnostic: finite effects of separate removals need not add. Final readout generation is excluded, so mismatch does not isolate non-additivity. This is not a measured joint removal or independent circuit extraction.')
(P/'FINITE_INTERACTION_ADDITIVITY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary))
