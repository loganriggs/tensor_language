"""Shared export/scoring for coefficient-metric experiments; native inputs explicit."""
import copy,torch
from shared_mixed_source_graph import component_scalars

def matrices(s):
 raw=torch.einsum('ir,ro,jr->oij',s['left_reader'],s['product_weights'],s['right_reader'])
 return (raw+raw.transpose(-1,-2))/2

def export(L,R,W,data,original):
 p=copy.deepcopy(original);s=p['shared_mixed']
 s['left_reader']=data['inverse_root']@L;s['right_reader']=data['inverse_root']@R
 s['product_weights']=W.T*data['scales'][None,:4];Qs=matrices(s)
 for o,Q in enumerate([q for pair in data['pairs'][:2] for q in pair['Qs']]):
  delta=Q-Qs[o];s['source_linear'][:,o]=2*delta@data['mu']
  s['source_bias'][o]=torch.trace(data['old_covariance']@delta)-data['mu']@delta@data['mu']
 return p

def score(p,data):
 ids=data['indices'];z=data['z'][ids];h=data['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
 phi=component_scalars(z,h,p);value=[]
 for j,pair in enumerate(data['pairs']):
  true=pair['truth'][ids];value.append(float((phi[:,j]-true).norm()/(true-true.mean()).norm()))
 s=p['shared_mixed'];hat=matrices(s);true=torch.stack([q for pair in data['pairs'][:2] for q in pair['Qs']])
 def jac(Q,lin,bias):
  reads=torch.einsum('ni,oij,nj->no',z,Q,z)+z@lin+bias
  grad=2*torch.einsum('oij,nj->noi',Q,z)+lin.T[None];out=[]
  for j,pair in enumerate(data['pairs'][:2]):
   a=(h@pair['a']-.5*reads[:,2*j])/scale-pair['alpha'];b=reads[:,2*j+1]/scale-pair['beta']
   out.append(-.5*(b/scale)[:,None]*grad[:,2*j]+(a/scale)[:,None]*grad[:,2*j+1])
  return out
 J=jac(hat,s['source_linear'],s['source_bias']);N=jac(true,torch.zeros_like(s['source_linear']),torch.zeros_like(s['source_bias']))
 return dict(per_mode_errors=value,euclidean_jacobian_errors=[float((j-n).norm()/n.norm()) for j,n in zip(J,N)])
