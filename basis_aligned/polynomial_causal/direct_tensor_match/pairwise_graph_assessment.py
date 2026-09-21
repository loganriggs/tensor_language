"""Reusable physical coefficient, execution and component checks for pairwise DAGs."""
import torch
from pairwise_reader_graph import expand,source_reads,price
from local_shared_reader_graph import decode,product_factors
from pack_reader_graph_artifacts import packed
from global_mixed_source_graph import score

def global_form(bundle):
 left=[];right=[];weights=[];linear=[];bias=[]
 for j in range(3):
  p=bundle[str(j)];L,R=product_factors(p['shared_reader'],p['product_indices']);left.append(L);right.append(R);W=L.new_zeros(L.shape[1],6);W[:,2*j:2*j+2]=p['product_weights'];weights.append(W)
  for k in ('a','b'):linear.append(p[k+'_linear']);bias.append(p[k+'_bias'])
 return dict(left_reader=torch.cat(left,1),right_reader=torch.cat(right,1),product_weights=torch.cat(weights),source_linear=torch.stack(linear,1),source_bias=torch.stack(bias),h_readers=torch.stack([bundle[str(j)]['h_reader'] for j in range(3)],1),alpha=torch.stack([bundle[str(j)]['alpha'] for j in range(3)]),beta=torch.stack([bundle[str(j)]['beta'] for j in range(3)]),residual_writer=bundle['0']['residual_writer'])

class Assessment:
 def __init__(self,data):
  self.data=data;self.Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);self.S=torch.linalg.inv(data['inverse_root']);self.I=torch.eye(self.Q.shape[-1],dtype=self.Q.dtype)
 @torch.no_grad()
 def correct(self,program):
  for j,p in enumerate(expand(program).values()):
   hat=decode(p)
   for name,true,H in zip(('a','b'),self.Q[2*j:2*j+2],hat):
    delta=true-H;program['pairs'][str(j)][name+'_linear']=2*delta@self.data['mu'];program['pairs'][str(j)][name+'_bias']=torch.trace(self.data['old_covariance']@delta)-self.data['mu']@delta@self.data['mu']
  return packed(program)
 @torch.no_grad()
 def assess(self,program):
  d=self.data;bundle=expand(program);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);errors=[]
  for A in (self.I,self.S):
   T=A@self.Q@A;E=A@(H-self.Q)@A;errors.append(float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt()))
  z=d['z'][d['indices']];g=global_form(bundle);dense=torch.einsum('ni,oij,nj->no',z,H,z)+z@g['source_linear']+g['source_bias'];replay=float((source_reads(z,program)-dense).norm()/dense.norm());assert replay<1e-8
  return dict(native_error=errors[0],covariance_error=errors[1],execution_replay=replay,**score(g,d),**price(program))
