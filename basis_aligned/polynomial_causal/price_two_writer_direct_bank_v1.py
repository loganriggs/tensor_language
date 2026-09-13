"""Matched local bank preparation: optimized28 versus direct19 products."""
from pathlib import Path
import torch,json,time,statistics
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_products
from two_writer_direct_bank_v1 import prepare
from two_writer_rational_basis_v1 import scalar_functions
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7140101);sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];rows=[]
 for batch in [1,8]:
  v=torch.randn(batch,7,1152,dtype=torch.float64);rho0=torch.ones(batch,1,dtype=torch.float64);beta=torch.tensor([.1,.2],dtype=torch.float64).expand(batch,-1);gamma=torch.tensor([[1.,.2],[.2,1.]],dtype=torch.float64)
  old=lambda:prepare_products(v,l,r,d)[1]
  new=lambda:prepare(v,l,r,d,rho0,beta,gamma)
  products=old();bank=new();a=torch.tensor(.3,dtype=torch.float64);b=torch.tensor(.5,dtype=torch.float64);rho=1-2*.1*a-2*.2*b+a*a+.4*a*b+b*b
  u=torch.stack([-a,-b,-a/rho,-b/rho,a*a/(2*rho),a*b/rho,b*b/(2*rho)])
  pairs=[(i,j) for i in range(7) for j in range(i,7)];weights=torch.stack([u[i]*u[j]*(1 if i==j else 2) for i,j in pairs]);ref=torch.einsum('k,bkd->bd',weights,products);fit=torch.einsum('k,bkd->bd',scalar_functions(a,b,rho),bank);error=float((fit-ref).norm()/ref.norm());assert error<1e-10
  times=[[],[]];old();new()
  for trial in range(7):
   for idx in ([0,1] if trial%2==0 else [1,0]):
    tic=time.perf_counter();[old,new][idx]();times[idx].append(time.perf_counter()-tic)
  oldtime,newtime=[statistics.median(x) for x in times];rows.append(dict(batch=batch,old_seconds=oldtime,new_seconds=newtime,speedup=oldtime/newtime,relative_error=error))
 result={'pred_a':all(x['relative_error']<1e-10 for x in rows),'pred_b':all(x['speedup']>=1.1 for x in rows),'rows':rows,'old_output_scalars_per_context':28*1152,'new_output_scalars_per_context':19*1152,'scope':'FP64 two-thread CPU, actualMLP10weights/synthetic vectorbanks. Both project7vectorsonce. Localbankpreparation only; no upstreambasis generation, original two-path semantics, native suffix or wholemodel savings.'}
 (P/'TWO_WRITER_DIRECT_BANK_V1_PRICE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
