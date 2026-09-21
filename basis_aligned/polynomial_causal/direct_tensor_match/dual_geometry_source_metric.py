"""Native-isotropic / covariance-shaped coefficient fitting in one native basis."""
import torch
from shared_private_metric import gram
class DualGeometrySourceMetric:
 def __init__(self,T,S,alpha,ridge=1e-10):
  self.T=T;self.S=S;self.alpha=alpha;self.ridge=ridge
  self.targets=[T,torch.einsum('ai,oij,jb->oab',S,T,S)]
  self.energies=[t.square().sum((-1,-2)).reshape(3,2).sum(1) for t in self.targets]
 def loss(self,L,R,V,detach=True):
  m=L.shape[1];A=torch.cat([L,V],1);B=torch.cat([R,V],1);objects=[]
  for index,target in enumerate(self.targets):
   a,b=(A,B) if index==0 else (self.S@A,self.S@B)
   objects.append((gram(a,b,a,b),torch.einsum('ir,oij,jr->ro',a,target,b)))
  weights=[self.alpha,1-self.alpha];rows=[];loss=0.
  for j in range(3):
   factor=sum(weight/energy[j] for weight,energy in zip(weights,self.energies))
   matrix=sum(weight*obj[0]/energy[j] for weight,obj,energy in zip(weights,objects,self.energies))
   rhs=sum(weight*obj[1][:,2*j:2*j+2]/energy[j] for weight,obj,energy in zip(weights,objects,self.energies))
   matrix=matrix+self.ridge*factor*torch.eye(len(matrix),dtype=matrix.dtype,device=matrix.device)
   wa=torch.linalg.solve(matrix[:m,:m],rhs[:m,0]);wa=torch.cat([wa,wa.new_zeros(len(matrix)-m)])
   n=len(matrix) if j==2 else m;wb=torch.linalg.solve(matrix[:n,:n],rhs[:n,1]);wb=torch.cat([wb,wb.new_zeros(len(matrix)-n)])
   w=torch.stack([wa,wb],1)
   if detach:w=w.detach()
   loss+= (1-2*(w*rhs).sum()+(w*(matrix@w)).sum())/3
   rows.append(w)
  full=torch.cat(rows,1).T
  return loss,full[:,:m],full[5,m:]
 def explicit(self,L,R,V,W,v):
  raw=torch.einsum('ir,or,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2
  one=torch.nn.functional.one_hot(torch.tensor(5,device=L.device),6).to(L);hat=hat+one[:,None,None]*((V*v)@V.T)[None]
  E=hat-self.T;errors=[E,torch.einsum('ai,oij,jb->oab',self.S,E,self.S)]
  reg=W.square().sum(1).reshape(3,2).sum(1)+torch.tensor([0.,0.,1.],dtype=L.dtype,device=L.device)*v.square().sum()
  return sum(weight*((e.square().sum((-1,-2)).reshape(3,2).sum(1)+self.ridge*reg)/energy).mean() for weight,e,energy in zip([self.alpha,1-self.alpha],errors,self.energies))
