"""Reproduce selected gauge search and export a hard four-interaction program."""
import json,math
from pathlib import Path
import torch
from core import Model,terms
from sweep import target_cases
from sparse_basis import gauge,coefficients,sparse_refits,dense_core

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;report=json.loads((p/'SPARSE_BASIS_SWEEP_V1.json').read_text());best=min(report['records'],key=lambda x:next(y for y in x['refits'] if y['budget']==4)['relative_gaussian_error'])
 r=json.loads((p/'TOY_SWEEP_V1.json').read_text());i=min((i for i,x in enumerate(r['records']) if x['case']=='sparse_tucker'),key=lambda i:r['records'][i]['relative_error']);state=torch.load(p/'TOY_SWEEP_V1.pt',weights_only=True,map_location='cpu')[i]['state'];target=next(x['target'] for x in target_cases() if x['name']=='sparse_tucker')
 torch.manual_seed(best['seed']);K=torch.nn.Parameter(torch.randn(3,3,dtype=torch.float64)*.05);J=torch.nn.Parameter(torch.randn(2,2,dtype=torch.float64)*.05);lr=best['lr']
 opt=torch.optim.Adam([K,J],lr=lr) if best['optimizer']=='adam' else torch.optim.Muon([K,J],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');value=float('inf');saved=None
 for step in range(3000):
  opt.zero_grad();P,W,G=gauge(state,K,J);loss=G.abs().sum()
  if float(loss.detach())<value:value=float(loss.detach());saved=(K.detach().clone(),J.detach().clone())
  loss.backward();opt.step()
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/3000)))
 with torch.no_grad():
  P,W,G=gauge(state,*saved);fit=sparse_refits(P,W,G,target,(4,))[0];sparse=torch.zeros_like(G).flatten();sparse[fit['support']]=torch.tensor(fit['coefficients'],dtype=torch.float64);sparse=sparse.reshape_as(G)
  # Independent native-index tensor contraction, rather than the training monomial map.
  T=torch.einsum('oa,apq,pi,qj->oij',W,dense_core(sparse,3),P,P);reference=dense_core(target,6);D=T-reference
  energy=lambda z:2*z.square().sum()+z.diagonal(dim1=-2,dim2=-1).sum(-1).square().sum()
  error=float((energy(D)/energy(reference)).sqrt());assert abs(error-fit['relative_gaussian_error'])<1e-10
  assert abs(error-next(x for x in best['refits'] if x['budget']==4)['relative_gaussian_error'])<1e-8
  x=torch.randn(64,6,dtype=torch.float64)*2+1;s=x@P.T;ij=list(terms(3,2));pred=torch.zeros(64,3,dtype=torch.float64)
  for flat,c in zip(fit['support'],fit['coefficients']):
   a,pair=divmod(flat,6);i,j=ij[pair];pred+=c*(s[:,i]*s[:,j])[:,None]*W[:,a]
  direct=torch.einsum('bi,oij,bj->bo',x,T,x);replay=float((pred-direct).abs().max());assert replay<1e-10
 result=dict(input_features=P.tolist(),output_writers=W.tolist(),pair_order=ij,support=fit['support'],coefficients=fit['coefficients'],gaussian_error=error,independent_execution_error=replay,float_values=28,core_interactions=4,source_optimizer=best['optimizer'],source_lr=lr,source_seed=best['seed'],scope='Hard sparse program for planted quadratic, exact coefficient-only refit after gauge search; not a native-model semantic circuit.')
 out=p/'SPARSE_TUCKER_PROGRAM_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['gaussian_error','independent_execution_error','float_values','core_interactions']})
if __name__=='__main__':main()
