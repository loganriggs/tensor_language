"""General source-cubic projection with native QK gates and arbitrary positions.
Extends compiled_shared_head_v1 from selected shared-parent children to a full
mixed source-feature bank. Caller declares current/first source states and the
output map (physical O or folded consumer C O). No native state generation.
"""
import itertools
import torch
from shared_cubic_source_projection_v1 import atom_gram
from cubic_projected_function_compare_v1 import solve
from folded_normalized_router_v1 import rotary,EPS

def compile_head(components,mix,q1,k1,q2,k2,values,output):
 c=components.double();m=mix.double();g=m@atom_gram(c)@m.T;d=k1.shape[1]
 return dict(components=c.clone(),mix=m.clone(),dual=solve(g,m),q1=q1.clone(),k1=k1.clone(),q2=q2.clone(),k2=k2.clone(),
  atom_k1=torch.einsum('kd,rid->rik',k1.double(),c[:,:,:d]),atom_k2=torch.einsum('kd,rid->rik',k2.double(),c[:,:,:d]),
  atom_write=torch.einsum('ok,kz,riz->rio',output.double(),values.double(),c)/6)

def execute_pairs(query,source,rotation,p):
 dtype=p['q1'].dtype;current=source[:,:p['k1'].shape[1]].to(dtype)
 qa=query.to(dtype)@p['q1'].T;qb=query.to(dtype)@p['q2'].T
 ka=current@p['k1'].T;kb=current@p['k2'].T;width=qa.shape[-1]
 # Full original norms at the actual source/query state, never per-factor norms.
 norms=(qa.square().mean(-1)+EPS)*(qb.square().mean(-1)+EPS)*(ka.square().mean(-1)+EPS)*(kb.square().mean(-1)+EPS)
 gate=1/(width**2*norms.sqrt())
 phi=torch.einsum('nd,rid->nri',source.double(),p['components']).prod(-1)@p['mix'].T
 dual=phi@p['dual']
 r=rotation.double()
 if r.ndim==2:r=r.expand(len(query),-1,-1)
 a=torch.einsum('nk,nkl,ril->nri',qa.double(),r,p['atom_k1']);b=torch.einsum('nk,nkl,ril->nri',qb.double(),r,p['atom_k2'])
 result=source.new_zeros((len(query),p['atom_write'].shape[-1]),dtype=torch.float64)
 for i,j,k in itertools.permutations(range(3)):
  result+=torch.einsum('nr,nr,nr,ro->no',dual,a[:,:,i],b[:,:,j],p['atom_write'][:,k])
 return gate.double()[:,None]*result

def execute_sequence(current,first,p):
 """Causal source sum for a whole batch; explicit positions, no score cache.
 Reference implementation chunks by query position; optimize only after replay.
 """
 batch,length,dim=current.shape;width=p['q1'].shape[0];output=[]
 rotations=[rotary(t,width).to(current.device) for t in range(length)]
 for t in range(length):
  q=current[:,t,None,:].expand(-1,t+1,-1).reshape(-1,dim)
  s=torch.cat((current[:,:t+1],first[:,:t+1]),-1).reshape(batch*(t+1),-1)
  rr=torch.stack([rotations[t].T@rotations[j] for j in range(t+1)])[None].expand(batch,-1,-1,-1).reshape(batch*(t+1),width,width)
  output.append(execute_pairs(q,s,rr,p).reshape(batch,t+1,-1).sum(1))
 return torch.stack(output,1)
