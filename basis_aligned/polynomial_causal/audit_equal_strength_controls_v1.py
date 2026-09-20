"""Separate matching validity, absolute selectivity and relative improvement."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/expanded_controls_equal_strength_v1_result.json'
a=json.loads(source.read_text());records=[]
for r in a['records']:
    valid=r['strength_error']<=.05 and r['number_cosine']>=.95
    candidate=max(r['candidate_ratios'][4:]);baseline=max(r['baseline_ratios'][4:])
    records.append(dict(panel=r['panel'],family=r['family'],role=r['role'],matched=valid,strength_error=r['strength_error'],cosine=r['number_cosine'],amplitude=r['amplitude'],candidate_new_max=candidate,baseline_new_max=baseline,collateral_ratio=candidate/max(baseline,1e-30),candidate_old_max=max(r['candidate_ratios'][1:4]),baseline_old_max=max(r['baseline_ratios'][1:4])))
summary=[]
for role in ['subject','attractor']:
    rr=[r for r in records if r['role']==role];valid=[r for r in rr if r['matched']]
    summary.append(dict(role=role,count=len(rr),matched=len(valid),candidate_new_pass=sum(r['candidate_new_max']<=.1 for r in valid),baseline_new_pass=sum(r['baseline_new_max']<=.1 for r in valid),advantage25=sum(r['collateral_ratio']<=.75 for r in valid),any_improvement=sum(r['collateral_ratio']<1 for r in valid),median_collateral_ratio=float(np.median([r['collateral_ratio'] for r in valid])) if valid else None,worst_candidate=max(r['candidate_new_max'] for r in rr),worst_baseline=max(r['baseline_new_max'] for r in rr),min_cosine=min(r['cosine'] for r in rr)))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),predictions=a['predictions'],summary=summary,records=records,scope='Native outcome-calibrated control comparison; not predictive extraction or fresh OOD. Unmatched cells excluded from improvement credit, never counted as passes.')
(P/'EQUAL_STRENGTH_CONTROLS_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
