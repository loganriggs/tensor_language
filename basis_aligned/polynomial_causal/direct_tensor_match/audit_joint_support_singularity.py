"""Reproduce the first interrupted supported fit with explicit domain diagnostics."""
from pathlib import Path
import json,math,torch
from pairwise_product_toy_fixture import fixture
from pairwise_exact_supports import supports
from pairwise_reader_graph import GROUPS
from joint_orthogonal_private import JointOrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2)
T,bases,private,_,_=fixture(1);U=supports(T);known=[u.T@v for u,v in zip(U,list(bases)+list(private))]
metric=JointOrthogonalPrivateMetric(T,[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS]);expand=lambda ps:[u@v for u,v in zip(U,ps)]
def diagnostics(params):
 with torch.no_grad():
  ps=expand(params);out=[]
  for j,(a,b) in enumerate(GROUPS):
   S=torch.linalg.qr(torch.cat([ps[a],ps[b]],1),mode='reduced').Q;D=torch.linalg.qr(ps[3+j],mode='reduced').Q;W=S.T@D;e=torch.linalg.eigvalsh(W.T@W);out.append(float(1-e[-1]**2))
  return out
rng=torch.Generator().manual_seed(42000);params=[torch.nn.Parameter(torch.randn(v.shape,dtype=v.dtype,generator=rng)/v.shape[0]**.5) for v in known];opt=torch.optim.Adam(params,lr=.03);history=[];failure=None
for step in range(1801):
 opt.zero_grad()
 try:loss,states=metric.loss(expand(params))
 except ValueError as error:
  failure=dict(step=step,error=str(error),denominators=diagnostics(params));break
 if step%50==0:history.append(dict(step=step,loss=float(loss.detach()),denominators=diagnostics(params),core_norms=[float(s[3].norm()) for s in states]))
 if step==1800:break
 opt.param_groups[0]['lr']=.03*.5*(1+math.cos(math.pi*step/1800));loss.backward();opt.step()
out=dict(case=1,optimizer='adam',seed=0,history=history,failure=failure,oracle_denominators=diagnostics(known),scope='Deterministic reproduction of the interrupted first failed arm. Same target, initialization, schedule and singularity threshold; no repair applied.')
(P/'JOINT_SUPPORTS_SINGULARITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
