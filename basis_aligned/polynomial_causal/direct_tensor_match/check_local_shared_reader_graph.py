from pathlib import Path
import torch,json
from local_shared_reader_graph import factor_bundle,expand,select_blocks,decode,source_reads,price,refit_pair,product_factors
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for seed in range(5):
 g=torch.Generator().manual_seed(25000+seed);dtype=torch.float64;d=16;n=8;r=2+seed%3;k=4
 def rand(*shape):return torch.randn(shape,dtype=dtype,generator=g)
 U=torch.linalg.qr(rand(d,r)).Q;bundle={};truth=[];selections={};writer=rand(d)
 for j in range(3):
  reader=rand(d,n);reader[:,:k]=U@rand(r,k)
  # Four two-dimensional pencil blocks: two products per block.
  a=torch.arange(0,n,2).repeat_interleave(2);b=a+1;kind=torch.tensor([1,2]*(n//2));indices=torch.stack([a,b,kind]);p=dict(shared_reader=reader,product_indices=indices,product_weights=rand(n,2),a_linear=rand(d),b_linear=rand(d),a_bias=rand(),b_bias=rand(),h_reader=rand(d),alpha=rand(),beta=rand(),residual_writer=writer);bundle[str(j)]=p;truth.append(decode(p));selected,diagnostic=select_blocks(p,U,torch.eye(d,dtype=dtype),k);assert selected.tolist()==list(range(k));selections[str(j)]=selected
 q=factor_bundle(bundle,U,selections);expanded=expand(q);z=rand(21,d)
 direct=torch.cat([torch.einsum('ni,oij,nj->no',z,truth[j],z)+torch.stack([z@bundle[str(j)][s+'_linear']+bundle[str(j)][s+'_bias'] for s in ('a','b')],1) for j in range(3)],1)
 error=float((source_reads(z,q)-direct).norm()/direct.norm());assert error<1e-12
 coefficient=max(float((decode(expanded[str(j)])-truth[j]).norm()/truth[j].norm()) for j in range(3));assert coefficient<1e-12
 assert price(q)['projection_multiplications']==d*(r+3*(n-k))+3*r*k
 p0=expanded['0'];noise=rand(2,d,d);target=truth[0]+.03*(noise+noise.transpose(-1,-2));W,fitcheck=refit_pair(p0,target,torch.eye(d,dtype=dtype))
 L,R=product_factors(p0['shared_reader'],p0['product_indices']);norm=L.norm(dim=0)*R.norm(dim=0);atoms=torch.einsum('ir,jr->rij',L,R);atoms=(atoms+atoms.transpose(-1,-2))/2;design=atoms.flatten(1).T/norm[None]
 augmented=torch.cat([design,1e-6*torch.eye(n,dtype=dtype)],0);rhs=torch.cat([target.flatten(1).T,torch.zeros(n,2,dtype=dtype)],0);oracle=torch.linalg.lstsq(augmented,rhs,driver='gelsd').solution
 replay=float((design@(W*norm[:,None])-design@oracle).norm()/target.norm());assert replay<1e-8
 rows.append(dict(readout_dense_least_squares_replay=replay,seed=seed,shared_width=r,selected_per_pair=k,execution_replay=error,coefficient_replay=coefficient,price=price(q)))
(P/'LOCAL_SHARED_READER_PREFLIGHT_V1.json').write_text(json.dumps(dict(controls=rows),indent=2)+'\n');print('PASS five planted overlapping dictionaries, complete-block selection and literal cost')
