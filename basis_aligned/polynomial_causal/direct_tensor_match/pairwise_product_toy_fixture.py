import torch
from pairwise_reader_graph import GROUPS
from local_shared_reader_graph import product_factors
def fixture(case):
 rng=torch.Generator().manual_seed(35000+case);r=1+case%3;shared=2*r;n=2*shared+4;k=shared+2;d=n+3
 rand=lambda *s:torch.randn(*s,dtype=torch.float64,generator=rng)
 bases=[rand(d,r) for _ in range(3)];private=[rand(d,n-k) for _ in range(3)];maps=[rand(shared,k) for _ in range(3)]
 i=torch.arange(0,n,2).repeat_interleave(2);indices=torch.stack([i,i+1,torch.tensor([1,2]*(n//2))]);templates=[];targets=[]
 for j,(a,b) in enumerate(GROUPS):
  selected=torch.arange(k);other=torch.arange(k,n);templates.append(dict(shared_indices=selected,private_indices=other,product_indices=indices.clone()))
  reader=torch.cat([torch.cat([bases[a],bases[b]],1)@maps[j],private[j]],1);L,R=product_factors(reader,indices);W=rand(n,2);raw=torch.einsum('ir,ro,jr->oij',L,W,R);targets.append((raw+raw.transpose(-1,-2))/2)
 return torch.cat(targets),bases,private,maps,templates
