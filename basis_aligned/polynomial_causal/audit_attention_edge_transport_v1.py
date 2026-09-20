"""Separate local algebra, fixed-reader transport, and finite interaction attenuation."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/attention_mixed_edge_native_v1_result.json'
a=json.loads(source.read_text());rows=[]
for g in a['groups']:
    for r in g['records']:
        rows.append(dict(panel=g['panel'],template=g['template'],**r,transport_relative_error=r['fixed_reader_error']/max(r['native_removal_norm'],1e-30)))
failed=a['failed_cells']
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),instrument=a['predictions'],max_checks={k:max(c[k] for c in a['checks']) for k in a['checks'][0]},failed_cell_summary=dict(count=len(failed),halved=sum(r['remaining_ratio']<=.5 for r in failed),increased=sum(r['remaining_ratio']>1 for r in failed),median_remaining=float(np.median([r['remaining_ratio'] for r in failed])),range_remaining=[min(r['remaining_ratio'] for r in failed),max(r['remaining_ratio'] for r in failed)]),groups=[],records=rows)
for template in sorted({r['template'] for r in rows}):
    rr=[r for r in rows if r['template']==template]
    out['groups'].append(dict(template=template,count=len(rr),transport_error_median=float(np.median([r['transport_relative_error'] for r in rr])),transport_error_max=max(r['transport_relative_error'] for r in rr),removal_norm_min=min(r['native_removal_norm'] for r in rr),removal_norm_max=max(r['native_removal_norm'] for r in rr)))
(P/'ATTENTION_EDGE_TRANSPORT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
