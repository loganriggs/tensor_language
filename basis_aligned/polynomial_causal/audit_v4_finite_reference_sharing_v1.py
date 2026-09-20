"""Certified shared-amplitude baseline for the SAME finite-target tangent LP."""
from pathlib import Path
import json,torch,numpy as np
from shared_selective_source_lp import choose
from refined_native_sources import PARENTS
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
groups=torch.load(a/'v4_source_oracle_audit_v1_tensors.pt',weights_only=False,map_location='cpu')['groups'];choices=json.loads((p/'V4_FINITE_REFERENCE_CHOICES_V1.json').read_text())['records'];reference=np.array([0,0,1,1,1,0])[PARENTS];records=[]
for role in ['subject','attractor']:
 for template in [None,'close_to','next_to_subject']:
  gs=[];ts=[]
  for g in groups:
   if g['role']!=role or (template is not None and g['template']!=template):continue
   z=torch.einsum('bod,bkd->bok',g['reader'],g['sources']);z[:,0]*=g['orientation'][:,None];gs.append(z.numpy());ts.extend(next(c['targets'] for c in choices if all(c[k]==g[k] for k in ['panel','template','role'])))
  weights,check=choose(np.concatenate(gs),reference,target_effect=np.array(ts))
  records.append(dict(role=role,template=template,rows=len(ts),amplitudes=weights.tolist(),certificate=check))
out=dict(records=records,scope='Opened all-case per-row linear constraints with finite targets, stronger than family-aggregate native gates; NOT a lower bound on nonlinear shared-selector feasibility or interpretable circuits')
f=p/'V4_FINITE_REFERENCE_SHARING_V1.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n')
for r in records:print(r['role'],r['template'],r['certificate']['retention'])
