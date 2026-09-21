"""Independent ALS audit of the only neighborhood not ruled out by rank bounds.

Does not change the original screen gates or select using held-out data.
"""
from pathlib import Path
import json,torch,time
from joint_product_refactor import best_subset,reconstruct
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_JOINT_PRODUCT_ALS_AUDIT_V1.json';assert not out.exists()
previous=json.loads((p/'MIDPOINT_JOINT_PRODUCT_REFACTOR_V1.json').read_text());eligible=[g for g in previous['groups'] if g['unfolding_lower_bound']<=.75*g['baseline_relative_squared_error']]
e=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();n-=n.mean(0);m-=m.mean(0);S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
factors=[n@e['A']/len(n)**.5,m@e['B']/len(m)**.5,S@e['reduced_writers']];grams=[f.T@f for f in factors]
records=[];start=time.perf_counter()
for item in eligible:
 ix=torch.tensor(item['products']);roots=[]
 for g in grams:
  ev,U=torch.linalg.eigh(g[ix[:,None],ix[None,:]]);roots.append(ev.clamp_min(0).sqrt()[:,None]*U.T)
 core=torch.einsum('ik,jk,lk->ijl',*roots);den=core.square().sum();base=best_subset(core,roots[0].T,roots[1].T,6)
 gen=torch.Generator().manual_seed(261317);batch=16;rank=6
 a=torch.randn(batch,8,rank,dtype=torch.float64,generator=gen);b=torch.randn_like(a);c=torch.randn_like(a)
 # Deterministically seed ALL random tensors, including b/c.
 b=torch.randn(batch,8,rank,dtype=torch.float64,generator=gen);c=torch.randn(batch,8,rank,dtype=torch.float64,generator=gen)
 for j in range(8):
  noise=[0.,1e-4,1e-3,1e-2,.03,.1,.3,.5][j]
  for t,v in zip([a,b,c],[base[2].T,base[3].T,base[4]]):t[j]=v+noise*torch.randn(v.shape,dtype=v.dtype,generator=gen)*v.norm()/v.numel()**.5
 best=torch.full((batch,),float('inf'),dtype=torch.float64);at=torch.zeros(batch,dtype=torch.long);history=[]
 for step in range(2001):
  if step%10==0:
   pred=torch.einsum('bir,bjr,bkr->bijk',a,b,c);err=(pred-core).square().sum((1,2,3))/den
   changed=err<best;best=torch.minimum(best,err);at[changed]=step
   if step%500==0:history.append(dict(step=step,best=float(best.min()),random_best=float(best[8:].min())))
  if step==2000:break
  def solve(rhs,x,y):
   gram=torch.einsum('bdi,bdj->bij',x,x)*torch.einsum('bdi,bdj->bij',y,y)
   ridge=1e-12*gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-30)
   return torch.linalg.solve(gram+ridge[:,None,None]*torch.eye(rank,dtype=gram.dtype),rhs.transpose(1,2)).transpose(1,2)
  a=solve(torch.einsum('ijk,bjr,bkr->bir',core,b,c),b,c)
  b=solve(torch.einsum('ijk,bir,bkr->bjr',core,a,c),a,c)
  c=solve(torch.einsum('ijk,bir,bjr->bkr',core,a,b),a,b)
  # Balance each term to remove scaling gauge without changing its function.
  norms=torch.stack([v.norm(dim=1).clamp_min(1e-30) for v in [a,b,c]])
  target=norms.prod(0).pow(1/3)
  a*= (target/norms[0])[:,None,:];b*=(target/norms[1])[:,None,:];c*=(target/norms[2])[:,None,:]
 records.append(dict(group=item['group'],relative_squared_error_by_restart=best.tolist(),best_steps=at.tolist(),best=float(best.min()),error_vs_subset=float(best.min())/item['baseline_relative_squared_error'],random_best=float(best[8:].min()),history=history))
result=dict(records=records,screen_groups_with_threshold_ruled_out=[g['group'] for g in previous['groups'] if g not in eligible],seconds=time.perf_counter()-start,scope='16 ALS initializations, eight subset/noisy subset and eight independent random,2000sweeps. Independent algorithm, fixed teacher and metric. All fitting endpoints retained; no claim of globally optimal CP approximation.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
