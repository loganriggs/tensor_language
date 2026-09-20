"""Summarize preregistered transport errors without masking weak effects."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/attention_edge_transport_native_v1_result.json'
a=json.loads(source.read_text());rows=[dict(panel=g['panel'],template=g['template'],**r) for g in a['groups'] for r in g['records']]
summary=[]
for template in sorted({r['template'] for r in rows}):
    for radius in [.5,1.]:
        rr=[r for r in rows if r['template']==template and abs(r['coordinate'][0])==radius]
        modes={}
        for mode in ['baseline','joint','midpoint','integrated']:
            rel=[r['relative_errors'][mode][0] for r in rr]
            modes[mode]=dict(number_relative_median=float(np.median(rel)),number_relative_max=max(rel),number_pass_count=sum(r['errors'][mode][0]<=max(1e-8,.1*r['target_norms'][0]) for r in rr),absolute_error_max=max(max(r['errors'][mode]) for r in rr))
        summary.append(dict(template=template,radius=radius,count=len(rr),modes=modes))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),predictions=a['predictions'],summary=summary,logical_cost=a['counts'],limitations='Opened diagnostic. Actual joint states and native suffix derivative evaluations are required; no extracted circuit or new causal-sufficiency success.')
(P/'ATTENTION_TRANSPORT_CONTEXT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
