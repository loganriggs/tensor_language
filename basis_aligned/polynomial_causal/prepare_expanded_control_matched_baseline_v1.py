"""Freeze plain-source comparison strengths using only prior number responses."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'bilinear_quotient/circuits/followups'
prior_path=A/'source_ood_v2_result.json';prior=json.loads(prior_path.read_text())
control_path=A/'expanded_source_controls_v1_result.json';controls=json.loads(control_path.read_text())
reference=np.array([0.,0.,1.,1.,1.,0.]);records=[]
for target in controls['records']:
    panel,role,family=target['panel'],target['role'],target['family'];template=family.split('|')[0]
    c=next(c for c in prior['contexts'] if c['panel']==panel and c['role']==role and c['template']==template)
    rows=[r for r in json.loads((P/f'SOURCE_OOD_V2_{panel.upper()}_ROWS.json').read_text()) if r['template']==template]
    ids=[i for i,r in enumerate(rows) if r['family']==family]
    g=np.asarray(c['gradient'])[ids,0];h=np.asarray(c['hessian'])[ids,0]
    linear=-g@reference;quadratic=-.5*np.einsum('i,bij,j->b',reference,h,reference)
    desired=target['budget']
    coeff=[float(quadratic@quadratic),float(2*linear@quadratic),float(linear@linear),0.,-desired**2]
    roots=np.roots(np.trim_zeros(coeff,'f'));legal=sorted(float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<=z.real<=1)
    alpha=legal[0] if legal else min([0.,1.],key=lambda t:abs(np.linalg.norm(linear*t+quadratic*t*t)-desired))
    predicted=float(np.linalg.norm(linear*alpha+quadratic*alpha**2))
    records.append(dict(panel=panel,role=role,family=family,amplitude=alpha,matched_quadratic_root=bool(legal),target_number_norm=desired,predicted_number_norm=predicted,predicted_relative_mismatch=abs(predicted-desired)/desired,reference=reference.tolist()))
out=dict(source_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),target_receipt_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),records=records,scope='Match number-effect norm using frozen prior quadratic responses only. Native strength must be within5%before comparing collateral. No new-control optimization; unmatched cases remain explicit. No native baseline control outcomes measured.')
(P/'EXPANDED_CONTROL_MATCHED_BASELINE_V1_BINDING.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(count=len(records),matched=sum(r['matched_quadratic_root'] for r in records),range=[min(r['amplitude'] for r in records),max(r['amplitude'] for r in records)],worst_predicted_mismatch=max(r['predicted_relative_mismatch'] for r in records))))
