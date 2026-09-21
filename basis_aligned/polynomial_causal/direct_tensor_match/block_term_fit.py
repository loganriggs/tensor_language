"""Joint rank-(1,L,L) fitting; compare complete blocks, not internal product gauges."""
import torch,json,time,itertools
from pathlib import Path
P=Path(__file__).resolve().parent

def evaluate(factors):
 W,A,B=factors
 return torch.einsum('og,igr,jgr->oij',W,A,B)

def fit(target,groups,width,seed,lr,steps,initial=None):
 gen=torch.Generator().manual_seed(seed)
 if initial is None:
  W=torch.randn(target.shape[0],groups,generator=gen,dtype=target.dtype);W=W/W.norm(dim=0)/groups**.5;A=torch.randn(target.shape[1],groups,width,generator=gen,dtype=target.dtype);B=torch.randn(target.shape[2],groups,width,generator=gen,dtype=target.dtype);A=A/A.norm(dim=0)/width**.25;B=B/B.norm(dim=0)/width**.25;initial=[W,A,B]
 params=[v.clone().requires_grad_() for v in initial];opt=torch.optim.Adam(params,lr=lr);best=float('inf');saved=None
 for step in range(steps):
  opt.zero_grad();loss=(evaluate(params)-target).square().sum();value=float(loss.detach())
  if value<best:best=value;saved=[x.detach().clone() for x in params]
  loss.backward();opt.step()
 return dict(error=best**.5,factors=saved)

def block_cosines(first,second):
 w,a,b=first;v,c,d=second;blocks=torch.einsum('igr,jgr->gij',a,b).flatten(1);others=torch.einsum('igr,jgr->gij',c,d).flatten(1);corr=((w/w.norm(dim=0)).T@(v/v.norm(dim=0)))*((blocks/blocks.norm(dim=1)[:,None])@(others/others.norm(dim=1)[:,None]).T);g=len(blocks);perm=max(itertools.permutations(range(g)),key=lambda p:sum(float(corr[i,p[i]]) for i in range(g)));return [float(corr[i,perm[i]]) for i in range(g)]

def toys():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);gen=torch.Generator().manual_seed(261146);rand=lambda *s:torch.randn(*s,generator=gen);cases={};A=torch.zeros(8,3,2);B=A.clone()
 for g in range(3):A[2*g:2*g+2,g,:]=torch.eye(2);B[2*g:2*g+2,g,:]=torch.eye(2)
 cases['separate_blocks']=(rand(4,3),A,B);cases['generic_dense']=(rand(4,3),rand(8,3,2),rand(8,3,2));w=rand(4,3);w[:,1]=w[:,0]+.2*w[:,1];cases['near_collinear_outputs']=(w,rand(8,3,2),rand(8,3,2));a=rand(8,1,2).repeat(1,3,1);cases['shared_input_space']=(rand(4,3),a,rand(8,3,2));cases['proportional_outputs']=(rand(4,1).repeat(1,3),rand(8,3,2),rand(8,3,2));records=[];start=time.perf_counter()
 for name,f in cases.items():
  T=evaluate(f);scale=T.norm();T=T/scale;teacher=[f[0]/scale,f[1],f[2]];oracle=float((evaluate(teacher)-T).norm());assert oracle<1e-12;fits=[]
  for seed in [0,1,2]:
   for lr in [.003,.01]:
    r=fit(T,3,2,261146+seed,lr,1000);fits.append(dict(seed=seed,lr=lr,error=r['error'],block_cosines=block_cosines(teacher,r['factors'])))
  records.append(dict(case=name,oracle=oracle,fits=fits));print(name,min(x['error'] for x in fits),flush=True)
 result=dict(records=records,seconds=time.perf_counter()-start,scope='Five planted rank-(1,2,2) structures. Shared input/proportional output controls deliberately violate generic identification; low fiterror is not blockrecovery.')
 out=P/'BLOCK_TERM_TOYS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':toys()
