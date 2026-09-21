from pathlib import Path
import json,time,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from pairwise_exact_supports import supports
from pencil_shared_discovery import discover
from joint_orthogonal_private import JointOrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[];start=time.monotonic()
for case in range(5):
 clean,bases,private,_,_=fixture(case);rng=torch.Generator().manual_seed(48000+case);noise=torch.randn(clean.shape,dtype=clean.dtype,generator=rng);noise=(noise+noise.transpose(-1,-2))/2
 for j in range(3):noise[2*j:2*j+2]*=clean[2*j:2*j+2].norm()/noise[2*j:2*j+2].norm()
 for level in (1e-10,1e-8,1e-6,1e-4):
  target=clean+level*noise;row=dict(case=case,noise=level)
  try:
   support=supports(target);row['pair_support_dimensions']=[u.shape[1] for u in support[3:]]
   if all(u.shape[1]==target.shape[-1] for u in support[3:]):
    row.update(unique=False,passed=False,reason='Full input supports make intersection filter vacuous; enumeration skipped');rows.append(row);continue
   found=discover(target,bases[0].shape[1]);row['unique']=found['success']
   if found['success']:
    params=found['bases']+found['private'];metric=JointOrthogonalPrivateMetric(target,[torch.cat([params[a],params[b]],1) for a,b in GROUPS]);error=float(metric.loss(params,dense=True)[0].sqrt());spans=[]
    for actual,true in zip(found['bases'],bases):
     q=torch.linalg.qr(true,mode='reduced').Q;spans.append(float((actual@actual.T-q@q.T).norm()))
    row.update(error=error,projector_errors=spans,passed=error<=10*level and max(spans)<=100*level)
   else:row.update(passed=False,reason=found['reason'],candidate_counts=[s['candidates'] for s in found['stats']])
  except (ValueError,RuntimeError,AssertionError) as error:row.update(unique=False,passed=False,failure=str(error))
  rows.append(row)
summary={str(level):dict(unique=sum(r.get('unique',False) for r in rows if r['noise']==level),passed=sum(r['passed'] for r in rows if r['noise']==level)) for level in (1e-10,1e-8,1e-6,1e-4)}
out=dict(records=rows,summary=summary,registered_small_noise_pass=summary['1e-08']['passed']>=4,seconds=time.monotonic()-start,scope='Fixed symmetric full-input perturbation, scaled separately to each pair target. Support and intersection thresholds unchanged. Reconstruction scored against noisy target, recovered subspaces against clean planted ones. No native robustness implication.')
(P/'PENCIL_SHARED_NOISE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
