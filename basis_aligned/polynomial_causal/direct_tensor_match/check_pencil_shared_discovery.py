from pathlib import Path
import json,time,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from pencil_shared_discovery import discover
from joint_orthogonal_private import JointOrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[];start=time.monotonic()
for case in range(5):
 T,bases,private,_,_=fixture(case);r=bases[0].shape[1]
 try:
  found=discover(T,r);row=dict(case=case,success=found['success'],stats=found['stats'])
  if found['success']:
   params=found['bases']+found['private'];metric=JointOrthogonalPrivateMetric(T,[torch.cat([params[a],params[b]],1) for a,b in GROUPS]);error=float(metric.loss(params,dense=True)[0].sqrt());span=[]
   for true,actual in zip(bases,found['bases']):
    q=torch.linalg.qr(true,mode='reduced').Q;span.append(float((q@q.T-actual@actual.T).norm()))
   row.update(coefficient_error=error,shared_projector_errors=span,recovery_pass=error<1e-8 and max(span)<1e-6)
  else:row.update(reason=found['reason'],recovery_pass=False)
 except (ValueError,RuntimeError,AssertionError) as error:row=dict(case=case,success=False,recovery_pass=False,failure=str(error))
 rows.append(row);print(json.dumps(row),flush=True)
out=dict(records=rows,all_recovered=all(r['recovery_pass'] for r in rows),seconds=time.monotonic()-start,scope='Same five planted targets. Given architecture widths; selection uses only target pencils and range intersections, never planted directions. Uniqueness only within enumerated simple block candidates; no native noise or broad identifiability claim.')
(P/'PENCIL_SHARED_DISCOVERY_V1.json').write_text(json.dumps(out,indent=2)+'\n')
