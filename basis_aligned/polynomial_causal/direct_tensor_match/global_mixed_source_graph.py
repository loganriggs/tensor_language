"""All six native source reads share one mixed-product dictionary."""
import torch

def source_reads(z,p):
 return ((z@p['left_reader'])*(z@p['right_reader']))@p['product_weights']+z@p['source_linear']+p['source_bias']

def component_scalars(z,h,p):
 reads=source_reads(z,p);s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[...,None]
 return ((h@p['h_readers']-.5*reads[...,::2])/s-p['alpha'])*(reads[...,1::2]/s-p['beta'])

def export(L,R,W,d):
 left=d['inverse_root']@L;right=d['inverse_root']@R;weights=W.T*d['scales'][None]
 raw=torch.einsum('ir,ro,jr->oij',left,weights,right);Qs=(raw+raw.transpose(-1,-2))/2;linear=[];bias=[]
 for o,Q in enumerate([q for pair in d['pairs'] for q in pair['Qs']]):
  delta=Q-Qs[o];linear.append(2*delta@d['mu']);bias.append(torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu'])
 return dict(left_reader=left,right_reader=right,product_weights=weights,source_linear=torch.stack(linear,1),source_bias=torch.stack(bias),h_readers=torch.stack([p['a'] for p in d['pairs']],1),alpha=torch.stack([p['alpha'] for p in d['pairs']]),beta=torch.stack([p['beta'] for p in d['pairs']]),residual_writer=d['residual_writer'])


def score(p,d):
 ids=d['indices'];z=d['z'][ids];h=d['h'][ids]
 scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
 phi=component_scalars(z,h,p)
 values=[float((phi[:,j]-pair['truth'][ids]).norm()/(pair['truth'][ids]-pair['truth'][ids].mean()).norm()) for j,pair in enumerate(d['pairs'])]
 raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);hat=(raw+raw.transpose(-1,-2))/2
 true=torch.stack([q for pair in d['pairs'] for q in pair['Qs']])
 def jac(Q,lin,bias):
  reads=torch.einsum('ni,oij,nj->no',z,Q,z)+z@lin+bias
  grad=2*torch.einsum('oij,nj->noi',Q,z)+lin.T[None]
  out=[]
  for j,pair in enumerate(d['pairs']):
   a=(h@pair['a']-.5*reads[:,2*j])/scale-pair['alpha'];b=reads[:,2*j+1]/scale-pair['beta']
   out.append(-.5*(b/scale)[:,None]*grad[:,2*j]+(a/scale)[:,None]*grad[:,2*j+1])
  return out
 J=jac(hat,p['source_linear'],p['source_bias']);N=jac(true,torch.zeros_like(p['source_linear']),torch.zeros_like(p['source_bias']))
 return dict(per_mode_errors=values,euclidean_jacobian_errors=[float((j-n).norm()/n.norm()) for j,n in zip(J,N)])
