"""Compare structural sharing constraints using saved derivatives only."""
import json,hashlib
from pathlib import Path
import numpy as np
from shared_selective_source_lp import choose
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/shared_selective_sources_v1_result.json'
a=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.]);records=[]
for group in a['gradients']:
    if group['panel']!='opposite':continue
    role,template=group['role'],group['template'];g=np.asarray(group['gradient']);weights,check=choose(g,reference)
    held=next(v for v in a['gradients'] if v['panel']=='congruent' and v['role']==role and v['template']==template);h=np.asarray(held['gradient']);baseline=-h[:,0]@reference;number=-h[:,0]@weights
    retained=number*np.sign(baseline)/np.maximum(abs(baseline),1e-10)
    controls=np.abs(np.einsum('noi,i->no',h[:,1:],weights))/np.maximum(abs(baseline),1e-10)[:,None]
    records.append(dict(role=role,template=template,weights=weights.tolist(),fit_certificate=check,held_min_retention=float(retained.min()),held_max_control_reference_ratio=float(controls.max()),held_linear_pass=bool(np.all(retained>=.8) and np.all(controls<=.08))))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),coefficients=24,records=records,scope='Opened derivative-only structural dictionary test: fit opposite panel, held congruent. No native effects evaluated for these new weights. Certificates bound only the registered first-order program.')
(P/'SHARED_SOURCE_GROUPING_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
