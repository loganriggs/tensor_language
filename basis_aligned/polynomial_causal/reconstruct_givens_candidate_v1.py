"""Reconstruct the previously selected seed61332/round10; no new search choice."""
import torch

def reconstruct(t,budget=None):
 bases=[]
 for axis in [0,2]:
  f=t.movedim(axis,0).reshape(t.shape[axis],-1);bases.append(torch.linalg.eigh(f@f.T).eigenvectors)
 core=torch.einsum('op,oia,ab->pib',bases[0],t,bases[1]);initial=core.movedim(1,0).reshape(1152,-1).contiguous();scale=initial.square().mean().sqrt();x=initial/scale;total=float(x.square().sum());gen=torch.Generator().manual_seed(61332);rotations=[]
 for step in range(10):
  vals=x.flatten().square().sort(descending=True).values;k=int(torch.searchsorted(vals.cumsum(0),.99*total))+1;threshold=vals[k-1]
  perm=torch.randperm(1152,generator=gen);left,right=perm[::2],perm[1::2];a,b=x[left],x[right];before=(a.square().clamp_max(threshold)+b.square().clamp_max(threshold)).sum(-1);best=before.clone();angle_best=torch.zeros(576,dtype=x.dtype)
  for angle in torch.linspace(-torch.pi/4,torch.pi/4,17,dtype=x.dtype):
   co,si=angle.cos(),angle.sin();aa=co*a+si*b;bb=-si*a+co*b;value=(aa.square().clamp_max(threshold)+bb.square().clamp_max(threshold)).sum(-1);improved=value<best;best[improved]=value[improved];angle_best[improved]=angle
  accepted=(before-best)>3*threshold;angles=torch.where(accepted,angle_best,0)[:,None];co,si=angles.cos(),angles.sin();x[left]=co*a+si*b;x[right]=-si*a+co*b;rotations.append((left[accepted],right[accepted],co[accepted],si[accepted]))
 vals,order=x.flatten().square().sort(descending=True);k=int(torch.searchsorted(vals.cumsum(0),.99*total))+1;count=sum(len(r[0]) for r in rotations);assert (k,count)==(1122633,2333)
 if budget is not None:
  k=min(x.numel(),(budget-4*sum(b.numel() for b in bases)-(t.numel()+7)//8-12*count)//4)
  if k<0:raise ValueError('Budget cannot hold adapters and rotations')
 fitted=torch.zeros_like(x.flatten());fitted[order[:k]]=x.flatten()[order[:k]];fitted=fitted.reshape_as(x)
 for left,right,co,si in reversed(rotations):
  a,b=fitted[left].clone(),fitted[right].clone();fitted[left]=co*a-si*b;fitted[right]=si*a+co*b
 fitcore=(fitted*scale).reshape(1152,12,128).movedim(0,1);fit=torch.einsum('op,pib,ab->oia',bases[0],fitcore,bases[1])
 return fit,dict(seed=61332,rounds=10,entries=k,rotations=count,nominal_bytes=4*(sum(b.numel() for b in bases)+k)+(t.numel()+7)//8+12*count)
