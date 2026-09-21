from pathlib import Path
import json,torch
from overlap_varpro import OverlapMetric
from local_shared_reader_graph import product_factors
P=Path(__file__).parent;torch.set_num_threads(2)
def fixture(seed):
 g=torch.Generator().manual_seed(26000+seed);d=6+seed;r=1+seed%3;n=6;k=2 if seed<2 else 4;dtype=torch.float64
 def rand(*shape):return torch.randn(shape,dtype=dtype,generator=g)
 params=[rand(d,r)/d**.5];templates=[];targets=[]
 for j in range(3):
  selected=torch.arange(k);private=torch.arange(k,n)
  if seed==0:indices=torch.stack([torch.arange(n),torch.arange(n),torch.zeros(n,dtype=torch.int64)])
  else:
   i=torch.arange(0,n,2).repeat_interleave(2);indices=torch.stack([i,i+1,torch.tensor([1,2]*(n//2))])
  templates.append(dict(shared_indices=selected,private_indices=private,product_indices=indices));params.extend([rand(r,k)/r**.5,rand(d,n-k)/d**.5])
  reader=torch.zeros(d,n,dtype=dtype);reader[:,selected]=params[0]@params[-2];reader[:,private]=params[-1];L,R=product_factors(reader,indices);W=rand(n,2);raw=torch.einsum('ir,ro,jr->oij',L,W,R);targets.append((raw+raw.transpose(-1,-2))/2)
 return torch.cat(targets),templates,params

def main():
 records=[]
 for seed in range(5):
  targets,templates,truth=fixture(seed);metric=OverlapMetric(targets,templates)
  oracle=float(metric.loss(truth,dense=True)[0]);assert oracle<1e-7
  g=torch.Generator().manual_seed(27000+seed);params=[(p+.2*torch.randn(p.shape,dtype=p.dtype,generator=g)).requires_grad_() for p in truth]
  implicit=metric.loss(params)[0];dense=metric.loss(params,dense=True)[0];through=metric.loss(params,detach=False)[0]
  grads=[torch.autograd.grad(x,params) for x in (implicit,dense,through)]
  replay=max(float((a-b).norm()/(1+a.norm())) for mode in grads[1:] for a,b in zip(grads[0],mode))
  assert abs(float(implicit-dense))<1e-10 and replay<1e-8
  records.append(dict(seed=seed,oracle_regularized_objective=oracle,dense_loss_replay=abs(float(implicit-dense)),dense_and_envelope_gradient_replay=replay))
 (P/'OVERLAP_VARPRO_PREFLIGHT_V1.json').write_text(json.dumps(dict(controls=records),indent=2)+'\n');print('PASS five planted shared/private targets, dense losses, envelope gradients and exact-capacity controls')
if __name__=='__main__':main()
