"""Check whether reducing the frozen edit improves collateral per target strength."""
import json,hashlib
from pathlib import Path
import numpy as np
from two_site_composition_scoring import key
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/expanded_source_controls_v1_result.json'
a=json.loads(source.read_text());records=[]
for g in a['groups']:
    for role,axis in [('subject',0),('attractor',1)]:
        for family in dict.fromkeys(g['families']):
            ids=[i for i,f in enumerate(g['families']) if f==family]
            original=next(r for r in a['records'] if r['panel']==g['panel'] and r['family']==family and r['role']==role)
            for radius in [1.,.5]:
                coord=[0.,0.];coord[axis]=radius
                effect=np.asarray(g['targets'][key(coord)])[ids];norms=np.linalg.norm(effect,axis=0)
                records.append(dict(panel=g['panel'],family=family,role=role,radius=radius,number_norm=float(norms[0]),effect_norms=norms.tolist(),per_actual_number=(norms[4:]/max(norms[0],1e-30)).tolist(),per_scaled_frozen_number=(norms[4:]/(radius*original['budget'])).tolist()))
summary=[]
for role in ['subject','attractor']:
    for radius in [1.,.5]:
        rr=[r for r in records if r['role']==role and r['radius']==radius]
        summary.append(dict(role=role,radius=radius,count=len(rr),pass_actual_number=sum(max(r['per_actual_number'])<=.1 for r in rr),worst_actual_number=max(max(r['per_actual_number']) for r in rr),pass_scaled_budget=sum(max(r['per_scaled_frozen_number'])<=.1 for r in rr),worst_scaled_budget=max(max(r['per_scaled_frozen_number']) for r in rr)))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=summary,records=records,scope='Opened radius diagnostic on already frozen axes, not new optimization or fresh OOD. Semantic control validity still needs appropriate answer-preserving families.')
(P/'EXPANDED_CONTROL_RADIUS_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
