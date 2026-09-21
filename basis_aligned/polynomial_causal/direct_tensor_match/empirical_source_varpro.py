"""Full-calibration variable projection for coefficient + empirical source losses."""
import torch
from shared_private_metric import gram
class EmpiricalSourceVarpro:
 def __init__(self,T,x,K,Y,lam,ridge=1e-10):
  self.T=T;self.x=x;self.K=K;self.Y=Y;self.lam=lam;self.ridge=ridge
  self.ec=T.square().sum((-1,-2)).reshape(3,2).sum(1)
  self.ed=Y.square().mean(0).reshape(3,2).sum(1)
 def loss(self,L,R,V,detach=True):
  m=L.shape[1];A=torch.cat([L,V],1);B=torch.cat([R,V],1)
  G=gram(A,B,A,B);C=torch.einsum('ir,oij,jr->ro',A,self.T,B)
  X=(self.x@A)*(self.x@B)-(A*(self.K@B)).sum(0)
  H=X.T@X/len(X);D=X.T@self.Y/len(X);rows=[];loss=0.
  for j in range(3):
   matrix=G/self.ec[j]+self.lam*H/self.ed[j]+self.ridge/self.ec[j]*torch.eye(len(G),dtype=G.dtype,device=G.device)
   rhs=C[:,2*j:2*j+2]/self.ec[j]+self.lam*D[:,2*j:2*j+2]/self.ed[j]
   if j<2:
    w=torch.linalg.solve(matrix[:m,:m],rhs[:m]);w=torch.cat([w,torch.zeros((len(G)-m,2),dtype=G.dtype,device=G.device)])
   else:
    wa=torch.linalg.solve(matrix[:m,:m],rhs[:m,0]);wa=torch.cat([wa,torch.zeros(len(G)-m,dtype=G.dtype,device=G.device)]);wb=torch.linalg.solve(matrix,rhs[:,1]);w=torch.stack([wa,wb],1)
   if detach:w=w.detach()
   coefficient=1+(-2*(w*C[:,2*j:2*j+2]).sum()+(w*(G@w)).sum()+self.ridge*w.square().sum())/self.ec[j]
   empirical=1+(-2*(w*D[:,2*j:2*j+2]).sum()+(w*(H@w)).sum())/self.ed[j]
   loss=loss+(coefficient+self.lam*empirical)/3;rows.append(w)
  full=torch.cat(rows,1).T
  return loss,full[:,:m],full[5,m:]
 def explicit(self,L,R,V,W,v):
  raw=torch.einsum('ir,or,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2;one=torch.nn.functional.one_hot(torch.tensor(5,device=L.device),6).to(L);hat=hat+one[:,None,None]*((V*v)@V.T)[None]
  E=hat-self.T;ce=E.square().sum((-1,-2)).reshape(3,2).sum(1)
  q=torch.einsum('ni,oij,nj->no',self.x,hat,self.x)-torch.einsum('ij,oji->o',self.K,hat);de=(q-self.Y).square().mean(0).reshape(3,2).sum(1)
  reg=W.square().sum(1).reshape(3,2).sum(1)+torch.tensor([0.,0.,1.],device=L.device,dtype=L.dtype)*v.square().sum()
  return ((ce+self.ridge*reg)/self.ec+self.lam*de/self.ed).mean()
