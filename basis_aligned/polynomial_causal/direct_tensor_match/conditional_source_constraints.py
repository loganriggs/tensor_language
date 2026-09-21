"""Differentiable native fidelity constraints with only second source forms free."""
import torch
class ConditionalSourceConstraints:
 def __init__(self,data,parent_forms,baseline):
  self.d=data;self.parent=parent_forms;self.base=baseline
  self.true=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);self.S=torch.linalg.inv(data['inverse_root']);self.I=torch.eye(self.S.shape[0],dtype=self.S.dtype)
  ids=data['indices'];self.z=data['z'][ids];self.h=data['h'][ids];self.s=(self.h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
  with torch.no_grad():self.truth_jac=self.responses(self.true,False)[1]
 def responses(self,H,correct=True):
  d=self.d;z=self.z;s=self.s;delta=self.true-H
  lin=2*delta@d['mu'] if correct else torch.zeros_like(H[:,:,0]);bias=torch.einsum('ij,oij->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu']) if correct else torch.zeros(H.shape[0],dtype=H.dtype)
  reads=torch.einsum('ni,oij,nj->no',z,H,z)+z@lin.T+bias;grad=2*torch.einsum('oij,nj->noi',H,z)+lin[None];values=[];jac=[]
  for j,pair in enumerate(d['pairs']):
   a=(self.h@pair['a']-.5*reads[:,2*j])/s-pair['alpha'];b=reads[:,2*j+1]/s-pair['beta'];values.append(a*b);jac.append(-.5*(b/s)[:,None]*grad[:,2*j]+(a/s)[:,None]*grad[:,2*j+1])
  return values,jac
 def ratios(self,second):
  H=torch.stack([form for j in range(3) for form in (self.parent[2*j],second[j])]);values,jac=self.responses(H);out=[]
  for M,name in ((self.I,'native_error'),(self.S,'covariance_error')):
   E=M@(H-self.true)@M;T=M@self.true@M;out.append((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean()/(1.1*self.base[name])**2)
  for j,v in enumerate(values):
   truth=self.d['pairs'][j]['truth'][self.d['indices']];out.append((v-truth).square().sum()/((truth-truth.mean()).square().sum()*min(.15,1.1*self.base['per_mode_errors'][j])**2))
  for j,J in enumerate(jac):out.append((J-self.truth_jac[j]).square().sum()/(self.truth_jac[j].square().sum()*(1.1*self.base['euclidean_jacobian_errors'][j])**2))
  return torch.stack(out)
