"""Joint shared/private reader optimization with exact normalized product readouts."""
import torch
from local_shared_reader_graph import product_factors
from shared_private_metric import gram

class OverlapMetric:
 def __init__(self,targets,templates,ridge=1e-10):
  self.scales=targets.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt()
  self.targets=targets/self.scales.repeat_interleave(2)[:,None,None]
  self.templates=templates;self.ridge=ridge
 def reader(self,params,j):
  basis=params[0];a,b=params[1+2*j:3+2*j];t=self.templates[j]
  n=len(t['shared_indices'])+len(t['private_indices'])
  return basis.new_zeros(basis.shape[0],n).index_copy(1,t['shared_indices'],basis@a).index_copy(1,t['private_indices'],b)
 def loss(self,params,detach=True,dense=False):
  loss=0.;weights=[];readers=[]
  for j in range(3):
   reader=self.reader(params,j);readers.append(reader);L,R=product_factors(reader,self.templates[j]['product_indices'])
   ln=L.norm(dim=0).clamp_min(1e-20);rn=R.norm(dim=0).clamp_min(1e-20);L=L/ln;R=R/rn
   G=gram(L,R,L,R);target=self.targets[2*j:2*j+2];rhs=torch.einsum('ir,oij,jr->ro',L,target,R);matrix=G+self.ridge*torch.eye(len(G),dtype=G.dtype,device=G.device);W=torch.linalg.solve(matrix,rhs)
   if detach:W=W.detach()
   if dense:
    raw=torch.einsum('ir,ro,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2
    piece=(hat-target).square().sum()+self.ridge*W.square().sum()
   else:piece=1-2*(rhs*W).sum()+(W*(matrix@W)).sum()
   loss+=piece/3;weights.append(W/(ln*rn)[:,None]*self.scales[j])
  return loss,weights,readers

def parameters_from_program(program,transform):
 params=[transform@program['input_basis']]
 for j in range(3):
  p=program['pairs'][str(j)];params.extend([p['shared_map'].clone(),transform@p['private_reader']])
 # Exact basis-column gauge normalization; maps compensate, private paths unchanged.
 norm=params[0].norm(dim=0).clamp_min(1e-20);params[0]=params[0]/norm
 for j in range(3):params[1+2*j]=norm[:,None]*params[1+2*j]
 return params
