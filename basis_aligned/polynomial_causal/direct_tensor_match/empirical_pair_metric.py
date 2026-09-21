"""Fixed-feature coefficient + empirical polynomial metric, no lifted matrix materialized."""
import torch
from shared_private_metric import gram

def prepare(Ta,Tb,La,Ra,Lb,Rb,x,K,ca,cb,family):
 ea=Ta.square().sum()+Tb.square().sum()
 G=torch.block_diag(gram(La,Ra,La,Ra),gram(Lb,Rb,Lb,Rb))/ea
 rhs=torch.cat([torch.einsum('ir,ij,jr->r',La,Ta,Ra),torch.einsum('ir,ij,jr->r',Lb,Tb,Rb)])/ea
 Xa=(x@La)*(x@Ra)-(La*(K@Ra)).sum(0);Xb=(x@Lb)*(x@Rb)-(Lb*(K@Rb)).sum(0)
 Ya=torch.einsum('ni,ij,nj->n',x,Ta,x)-torch.trace(K@Ta);Yb=torch.einsum('ni,ij,nj->n',x,Tb,x)-torch.trace(K@Tb)
 if family=='source':
  norm=(Ya.square()+Yb.square()).mean();D=torch.block_diag(Xa,Xb)/norm.sqrt();target=torch.cat([Ya,Yb])/norm.sqrt();divisor=len(x)
 else:
  assert family=='downstream'
  D=torch.cat([ca[:,None]*Xa,cb[:,None]*Xb],1);target=ca*Ya+cb*Yb;divisor=len(x)
 return dict(G=G,rhs=rhs,empirical_gram=D.T@D/divisor,empirical_rhs=D.T@target/divisor,design=D,target=target,divisor=divisor,coefficient_energy=ea)

def solve(stats,lam):
 H=stats['G']+lam*stats['empirical_gram'];rhs=stats['rhs']+lam*stats['empirical_rhs'];ridge=1e-10*H.diag().mean();H=H+ridge*torch.eye(len(H),device=H.device,dtype=H.dtype)
 w=torch.linalg.solve(H,rhs);residual=float((H@w-rhs).norm()/rhs.norm())
 return w,ridge,residual
